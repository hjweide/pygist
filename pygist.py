import os
import cv2
import pickle
import pyflann
import subprocess
import numpy as np
from sklearn import svm


def compute_gist_descriptors(imgpaths, datafile='.descriptors.pickle'):
    if os.path.isfile(datafile):
        return pickle.load(open('.descriptors.pickle', 'rb'))

    descriptors = []
    for imgpath in imgpaths:
        if os.path.basename(imgpath).lower().endswith('.pgm'):
            c_args = ['gist/./compute_gist', imgpath]
            desc = subprocess.check_output(c_args).strip().split(' ')
            desc = map(float, desc)
            descriptors.append(desc)

    print 'Computed %d GIST descriptors' % len(descriptors)
    descriptors_arr = np.array(descriptors)
    pickle.dump(descriptors_arr, open('.descriptors.pickle', 'wb'))
    return descriptors_arr


def resize(imgpaths, outdir):
    if not os.path.isdir(outdir):
        print 'Created %s for storing resized images' % outdir
        os.makedirs(outdir)
    else:
        filenames = pickle.load(open(os.path.join(outdir, '.filenames.pickle'), 'rb'))
        return [os.path.join(outdir, f) for f in filenames]

    outpaths = []
    for imgpath in imgpaths:
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (32, 32))
        base = os.path.basename(imgpath)
        base_no_ext = os.path.splitext(base)[0]
        outpath = os.path.join(outdir, base_no_ext + '.pgm')
        outpaths.append(outpath)
        cv2.imwrite(outpath, img)

    print 'Resized %s images' % len(outpaths)
    pickle.dump(outpaths, open(os.path.join(outdir, '.filenames.pickle'), 'wb'))
    return outpaths


def train(imgpaths, labels, k=5, trfrac=0.8):
    labels = np.array(labels)
    resized_paths = resize(imgpaths, '/home/hendrik/ibeis/pygist/.pygist_resized')
    dataset = compute_gist_descriptors(resized_paths)

    assert np.shape(labels)[0] == np.shape(dataset)[0]
    #np.savetxt('pos_descriptors.txt', dataset[labels == 1], fmt='%.4f')
    #np.savetxt('neg_descriptors.txt', dataset[labels == -1], fmt='%.4f')

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
    temp_test = temp[N:]

    # centre the data to have zero mean
    centering_transform = np.mean(training_set, 0)
    training_set -= centering_transform
    test_set -= centering_transform

    # scale the data to have unity standard deviation
    scaling_transform = np.std(training_set, 0)
    training_set /= scaling_transform
    test_set /= scaling_transform

    training_set_size, test_set_size = np.shape(training_set)[0], np.shape(test_set)[0]
    print 'Using %s images for training and %s images for testing' % (training_set_size, test_set_size)
    print '\t %d positive training examples and %d negative training examples' % (np.shape(training_labels[training_labels == 1])[0], np.shape(training_labels[training_labels == -1])[0])
    
    flann = pyflann.FLANN()
    centroids = flann.kmeans(training_set[training_labels == 1], k)
    
    #np.set_printoptions(threshold=np.nan)
    positive_descriptor_assignments, _ = flann.nn(centroids, training_set[training_labels == 1], 1)
    
    print 'Assignment of positive descriptors to clusters:'
    print np.bincount(positive_descriptor_assignments)

    assigned_labels_train = np.zeros((np.shape(training_set)[0], k))
    assigned_labels_test = np.zeros((np.shape(test_set)[0], k))

    for i in range(0, k, 1):
        training_labels_copy = np.copy(training_labels)
        #training_labels_copy[training_labels == 1][positive_descriptor_assignments != i] = -1
        training_labels_copy[[a[positive_descriptor_assignments != i] for a in np.where(training_labels_copy == 1)]] = -1

        prior = (np.shape(training_labels_copy[training_labels_copy == 1])[0]) / float(np.shape(training_labels_copy[training_labels_copy == -1])[0])

        classifier = svm.SVC(kernel='rbf', class_weight={1: 1. / prior, -1: 1.})
        SVM = classifier.fit(training_set, training_labels_copy)
        predicted_train = classifier.predict(training_set)
        predicted_test = classifier.predict(test_set)

        assigned_labels_train[:, i] = predicted_train
        assigned_labels_test[:, i] = predicted_test

        error = 0.
        for j in range(0, training_set_size, 1):
            if training_labels_copy[j] != predicted_train[j]:
                error += 1.
        print 'Training error for classifier %d is %.2f (%d/%d)' % (i, error / training_set_size, int(error), training_set_size)

    error = 0.
    fp, fn = 0, 0
    for i in range(0, test_set_size, 1):
        predicted_label = 1 if 1 in assigned_labels_test[i, :] else -1
        #print assigned_labels_test[i, :], predicted_label, test_labels[i], temp_test[i]
        if predicted_label != test_labels[i]:
            error += 1.
        if predicted_label == 1 and test_labels[i] == -1:
            fp += 1
        elif predicted_label == -1 and test_labels[i] == 1:
            fn += 1

    print 'Test error is %.2f (%d/%d)' % (error / test_set_size, int(error), test_set_size)
    print 'False Positives: %d' % fp
    print 'False Negatives: %d' % fn


if __name__ == '__main__':
    target_dir = '/home/hendrik/ibeis/pygist/images'
    imgpaths = [os.path.join(target_dir, f) for f in os.listdir(target_dir)]
    labels = [1 if '2014' in f else -1 for f in imgpaths]
    #for i, l in zip(imgpaths, labels):
    #    print i, l
    train(imgpaths, labels, k=5)
