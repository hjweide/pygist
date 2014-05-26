#/bin/sh
mkdir build
cd build
cmake -G "Unix Makefiles" .. && make
cd ..

cp build/compute_gist pygist
