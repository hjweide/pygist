from __future__ import absolute_import, division, print_function


#set PATH=C:\Users\joncrall\code\pygist\gist\fftw;%PATH%

def install_ffw_mingw():
    import utool
    import os
    fftw_targz_src_url     = 'http://www.fftw.org/fftw-3.3.4.tar.gz'
    mingw_fftw_build64_url = 'ftp://ftp.fftw.org/pub/fftw/BUILD-MINGW64.sh'
    #utool.grab_zipped_url('ftp://ftp.fftw.org/pub/fftw/fftw-3.3.4-dll64.zip')
    #mingw_fftw_build32 = 'ftp://ftp.fftw.org/pub/fftw/BUILD-MINGW32.sh'
    fftw_src = utool.grab_zipped_url(fftw_targz_src_url)
    utool.view_directory(fftw_src)
    os.chdir(fftw_src)
    buildscript = utool.grab_file_url(mingw_fftw_build64_url, download_dir=fftw_src)

    'sh configure --with-our-malloc16 --with-windows-f77-mangling --enable-shared --disable-static --enable-threads --with-combined-threads --enable-portable-binary --enable-sse2 --with-incoming-stack-boundary=2'

    'make && make install'
    #os.system('sh ' + buildscript)
