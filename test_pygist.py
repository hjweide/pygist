#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import pygist
import utool
from itertools import izip


TEST_MODEL_URL  = 'https://dl.dropbox.com/s/mbqovlwsk2j1tws/.learned_model.pickle'
TEST_IMAGES_URL = 'https://dl.dropboxusercontent.com/s/of2s82ed4xf86m6/testdata.zip'


def test_pygist():
    print('[pygist] Testing pygist')
    # Ensure you have test data
    print('[pygist] Ensuring testdata')
    datafile       = utool.grab_file_url(TEST_MODEL_URL, appname='utool')
    test_image_dir = utool.grab_zipped_url(TEST_IMAGES_URL, appname='utool')
    imgpaths       = utool.list_images(test_image_dir, fullpath=True)   # test image paths
    outdir = utool.get_app_resource_dir('pygist')  # where to put results
    # Run pygist on test images
    print('[pygist] Running tests')
    test_results = pygist.test(imgpaths, outdir=outdir, datafile=datafile)
    # Print results
    target_results = [-1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1]
    assert target_results == target_results, 'results do not match'
    print('test_results = %r' % (test_results,))
    print(utool.list_str(list(izip(imgpaths, test_results))))
    return locals()


if __name__ == '__main__':
    test_locals = utool.run_test(test_pygist)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    exec(utool.ipython_execstr())
