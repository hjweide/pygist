import os
import cv2
import pickle
import pyflann
import subprocess
import numpy as np


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
        return [os.path.join(outdir, f) for f in os.listdir(outdir)]

    outpaths = []
    for imgpath in imgpaths:
        print imgpath
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (32, 32))
        base = os.path.basename(imgpath)
        base_no_ext = os.path.splitext(base)[0]
        outpath = os.path.join(outdir, base_no_ext + '.pgm')
        outpaths.append(outpath)
        cv2.imwrite(outpath, img)

    print 'Resized %s images' % len(outpaths)
    return outpaths


def train(imgpaths, labels, k=5, trfrac=0.8):
    labels = np.array(labels)
    resized_paths = resize(imgpaths, '/home/hendrik/ibeis/pygist/.pygist_resized')
    dataset = compute_gist_descriptors(resized_paths)

    assert np.shape(labels)[0] == np.shape(dataset)[0]

    state = np.random.get_state()
    np.random.shuffle(labels)
    np.random.set_state(state)
    np.random.shuffle(dataset)

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
    scaling_transform = np.mean(training_set, 0)
    training_set /= scaling_transform
    test_set /= scaling_transform

    training_set_size, test_set_size = np.shape(training_set)[0], np.shape(test_set)[0]
    print 'Using %s images for training and %s images for testing' % (training_set_size, test_set_size)
    print '\t %d positive training examples and %d negative training examples' % (np.shape(training_labels[training_labels == 1])[0], np.shape(training_labels[training_labels == -1])[0])
    
    flann = pyflann.FLANN()
    centroids = flann.kmeans(training_set[training_labels == 1], k)

    positive_descriptor_assignments, _ = flann.nn(centroids, training_set[training_labels == 1], 1)
    print positive_descriptor_assignments
    print 'Assignment of positive descriptors to clusters:'
    print np.bincount(positive_descriptor_assignments)


if __name__ == '__main__':
    target_dir = '/home/hendrik/ibeis/pygist/images'
    imgpaths = [os.path.join(target_dir, f) for f in os.listdir(target_dir)]
    labels = [1 if '2014' in f else -1 for f in imgpaths]
    
    train(imgpaths, labels)
