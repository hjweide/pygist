#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from utool.util_setup import setuptools_setup
from utool import util_cplat
import utool


def build_command():
    """ Build command run by utool.util_setup """
    if util_cplat.WIN32:
        fftw_win32_url = 'ftp://ftp.fftw.org/pub/fftw/fftw-3.3.4-dll32.zip'
        #fftw_win64_url = 'ftp://ftp.fftw.org/pub/fftw/fftw-3.3.4-dll64.zip'
        # Ensure you have everything to build on windows
        setup_dir = utool.dirname(__file__)
        fftw_dir = utool.grab_zipped_url(fftw_win32_url, download_dir=setup_dir)
        print(fftw_dir)
        util_cplat.shell('mingw_build.bat')
    else:
        util_cplat.shell('./unix_build.sh')


INSTALL_REQUIRES = [
    'scikit-learn >= 0.14.1',
]

if __name__ == '__main__':
    setuptools_setup(
        name='pygist',
        build_command=build_command,
        description=('Filters images using gist descriptors'),
        url='https://github.com/hjweide/pygist',
        author='Hendrik Weideman',
        author_email='weideh@rpi.edu',
        packages=['pygist'],
        install_requires=INSTALL_REQUIRES,
        package_data={'build': util_cplat.get_dynamic_lib_globstrs() +
                      ['compute_gist']},
        setup_fpath=__file__,
    )
