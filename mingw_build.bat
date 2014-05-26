mkdir build
cd build
cmake -G "MSYS Makefiles" .. && make
cd ..

cp fftw/libfftw3f-3.dll pygist
cp build/compute_gist.exe pygist
