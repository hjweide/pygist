#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from utool.util_setup import setuptools_setup
from utool import util_cplat, WIN32


if WIN32:
    'ftp://ftp.fftw.org/pub/fftw/fftw-3.3.4-dll64.zip'


def build_command():
    """ Build command run by utool.util_setup """
    if util_cplat.WIN32:
        util_cplat.shell('mingw_build.bat')
    else:
        util_cplat.shell('./unix_build.sh')


INSTALL_REQUIRES = [
    'detecttools >= 1.0.0.dev1',
    'sklearn >= 0.14.1',
]

if __name__ == '__main__':
    setuptools_setup(
        name='pygist',
        build_command=build_command,
        description=('Filters images using gist descriptors'),
        url='https://github.com/bluemellophone/pyrf',
        author='Hendrik Weideman',
        author_email='weideh@rpi.edu',
        packages=['pygist'],
        install_requires=INSTALL_REQUIRES,
        package_data={'build': util_cplat.get_dynamic_lib_globstrs()},
        setup_fpath=__file__,
    )
