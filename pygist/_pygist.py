from __future__ import absolute_import, division, print_function
import sys
import os
import cv2
import cPickle
import pyflann
import subprocess
import numpy as np
from os.path import join, isfile, basename, isdir, splitext, dirname, exists
from sklearn import svm
from . import model

#import utool
#utool.find_exe('compute_gist', path_hints=[dirname(__file__),
#                                           join(dirname(__file__), '../gist')])
#COMPUTE_GIST_EXE = 'gist/./compute_gist'
#COMPUTE_GIST_EXE = dirname(__file__) + '../gist/compute_gist'

COMPUTE_GIST_EXE = dirname(__file__) + '/compute_gist'
if sys.platform == 'win32':
    COMPUTE_GIST_EXE += '.exe'
if not exists(COMPUTE_GIST_EXE):
    raise AssertionError("cannot find compute_gist")


def compute_gist_descriptors(imgpaths, datafile='.descriptors.pickle'):
    """
    compute the gist descriptors for the given image paths
    reload the descriptors from disk if they already exist
    """
    if isfile(datafile):
        return cPickle.load(open(datafile, 'rb'))

    descriptors = []
    for imgpath in imgpaths:
        if basename(imgpath).lower().endswith('.ppm'):
            # launch the compute_gist C executable
            c_args = [COMPUTE_GIST_EXE, imgpath]
            desc = subprocess.check_output(c_args).strip().split(' ')
            desc = map(float, desc)
            descriptors.append(desc)

    print('Computed %d GIST descriptors' % len(descriptors))
    descriptors_arr = np.array(descriptors)
    # dump the descriptors so that we do not need to recompute them each time
    cPickle.dump(descriptors_arr, open(datafile, 'wb'))
    return descriptors_arr


def resize_images(imgpaths, outdir):
    """
    resize the images at the given image paths
    write them to a directory to avoid resizing them every time
    """
    print('[pygist] resizing images to %r' % outdir)
    if not isdir(outdir):
        print('Created %s for storing resized images' % outdir)
        os.makedirs(outdir)
    else:
        # need to store the filenames because Python does not always read files in the same order
        filenames = cPickle.load(open(join(outdir, '.filenames.pickle'), 'rb'))
        return [join(outdir, f) for f in filenames]

    outpaths = []
    for imgpath in imgpaths:
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (32, 32))
        # get the filename without its path and extension
        base = basename(imgpath)
        base_no_ext = splitext(base)[0]
        outpath = join(outdir, base_no_ext + '.ppm')
        outpaths.append(outpath)
        cv2.imwrite(outpath, img)

    print('Resized %s images' % len(outpaths))
    cPickle.dump(outpaths, open(join(outdir, '.filenames.pickle'), 'wb'))
    return outpaths


def train(imgpaths, labels, outmodel='.learned_model.pickle', k=5, trfrac=0.8):
    """
    imgpaths: full paths to the images that make up the dataset
    labels:   1 or -1 for positive and negative training images, respectively
    k:        how many classifiers to train, 5 is recommended
    trfrac:   the fraction of the dataset that is used for training
    """
    labels = np.array(labels)
    resized_paths = resize_images(imgpaths, '/home/hendrik/ibeis/pygist/.pygist_resized')
    dataset = compute_gist_descriptors(resized_paths)

    # each descriptor must have a label or there is a problem
    assert np.shape(labels)[0] == np.shape(dataset)[0], 'each descriptor must have a label'

    temp = np.array(imgpaths)
    state = np.random.get_state()
    np.random.shuffle(labels)
    np.random.set_state(state)
    np.random.shuffle(dataset)
    np.random.set_state(state)
    np.random.shuffle(temp)

    N = int(trfrac * np.shape(dataset)[0])

    training_set = dataset[:N]
    training_labels = labels[:N]

    test_set = dataset[N:]
    test_labels = labels[N:]

    # centre the data to have zero mean
    centering_transform = np.mean(training_set, 0)
    training_set -= centering_transform
    test_set -= centering_transform

    # scale the data to have unity standard deviation
    scaling_transform = np.std(training_set, 0)
    training_set /= scaling_transform
    test_set /= scaling_transform

    training_set_size, test_set_size = np.shape(training_set)[0], np.shape(test_set)[0]
    print('Using %s images for training and %s images for testing' % (training_set_size, test_set_size))
    print('\t %d positive training examples and %d negative training examples' % (np.shape(training_labels[training_labels == 1])[0], np.shape(training_labels[training_labels == -1])[0]))

    flann = pyflann.FLANN()
    # cluster only the positive training examples into k clusters
    centroids = flann.kmeans(training_set[training_labels == 1], k)
    positive_descriptor_assignments, _ = flann.nn(centroids, training_set[training_labels == 1], 1)

    # view the distribution of descriptors to clusters
    print('Assignment of positive descriptors to clusters:')
    print(np.bincount(positive_descriptor_assignments))

    # to store the results of each classifier's output
    assigned_labels_train = np.zeros((np.shape(training_set)[0], k))
    assigned_labels_test = np.zeros((np.shape(test_set)[0], k))

    classifiers = []
    for i in xrange(0, k, 1):
        training_labels_copy = np.copy(training_labels)
        # use the positive training descriptors that do not belong to this cluster as negatives
        training_labels_copy[[a[positive_descriptor_assignments != i] for a in np.where(training_labels_copy == 1)]] = -1

        # compensate for the many negative training examples we created
        prior = (np.shape(training_labels_copy[training_labels_copy == 1])[0]) / float(np.shape(training_labels_copy[training_labels_copy == -1])[0])
        classifier = svm.SVC(kernel='rbf', class_weight={1: 1. / prior, -1: 1.})

        # fit the training set
        classifier.fit(training_set, training_labels_copy)
        classifiers.append(classifier)

        predicted_train = classifier.predict(training_set)
        assigned_labels_train[:, i] = predicted_train

        if N < np.shape(dataset)[0]:
            predicted_test = classifier.predict(test_set)
            assigned_labels_test[:, i] = predicted_test

        error = 0.
        for j in xrange(0, training_set_size, 1):
            if training_labels_copy[j] != predicted_train[j]:
                error += 1.
        print('Training error for classifier %d is %.2f (%d/%d)' % (i, error / training_set_size, int(error), training_set_size))

    # only evaluate the test set if one was given
    if N < np.shape(dataset)[0]:
        error = 0.
        fp, fn = 0, 0
        for i in xrange(0, test_set_size, 1):
            # a datapoint is positive if one of the classifers classified it as positive
            predicted_label = 1 if 1 in assigned_labels_test[i, :] else -1
            if predicted_label != test_labels[i]:
                error += 1.
            if predicted_label == 1 and test_labels[i] == -1:
                fp += 1
            elif predicted_label == -1 and test_labels[i] == 1:
                fn += 1

        print('Test error is %.2f (%d/%d)' % (error / test_set_size, int(error), test_set_size))
        print('False Positives: %d' % fp)
        print('False Negatives: %d' % fn)

    # save this model to a file to load during testing
    print('Trained model dumped to %s' % outmodel)
    learned_model = model.Model(classifiers, centering_transform, scaling_transform)
    cPickle.dump(learned_model, open(outmodel, 'wb'))


def test(imgpaths, outdir='.', datafile='.learned_model.pickle'):
    """
    imgpaths: full paths to the images that make up the dataset
    outdir:   directory where the resized images will be saved
    datafile: the file from which the pre-trained model should be loaded (if it exists)
    """
    test_resized_dir = join(outdir, 'pygist_resized_test')
    test_datafile = join(outdir, '.pygist_descriptors_test.pickle')

    # Ensure model descriptors have been trained
    if isfile(datafile):
        print('Reading: %s' % datafile)
        import sys
        sys.modules['model'] = model  # hack
        learned_model = cPickle.load(open(datafile, 'rb'))
    else:
        print('No trained model "%s" detected!' % datafile)
        return None

    # Ensure images have been resized
    resized_paths = resize_images(imgpaths, test_resized_dir)

    # Ensure test images have gist descriptors
    if not isfile(test_datafile):
        test_set = compute_gist_descriptors(resized_paths, datafile=test_datafile)
    else:
        test_set = cPickle.load(open(test_datafile, 'rb'))

    test_set -= learned_model.center
    test_set /= learned_model.scale

    k = len(learned_model.classifiers)
    test_set_size = np.shape(test_set)[0]
    assigned_labels_test = np.zeros((test_set_size, k))
    print('Stage 1: pre-classifying data using %d classifiers...' % k)
    for i, classifier in enumerate(learned_model.classifiers):
        predicted_test = classifier.predict(test_set)
        assigned_labels_test[:, i] = predicted_test

    predicted = []
    print('Stage 2: final classification on %d datapoints...' % test_set_size)
    for i in xrange(0, test_set_size, 1):
        predicted.append(1 if 1 in assigned_labels_test[i, :] else -1)

    return predicted


#if __name__ == '__main__':
#    target_dir = '/home/hendrik/ibeis/pygist/images'
#    imgpaths = [join(target_dir, f) for f in os.listdir(target_dir)]
#    labels = [1 if '2014' in f else -1 for f in imgpaths]
#    for i, l in zip(imgpaths, labels):
#        print i, l
#    train(imgpaths, labels, k=5)
#    target_dir = '/home/hendrik/ibeis/pygist/test'
