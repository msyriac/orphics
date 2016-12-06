#!/bin/bash
rm bin/deg2healpix.*
gcc -fPIC -c src/deg2healpix.c -I$HEALPIX/include
gcc -O3 -fPIC -shared -o deg2healpix.so deg2healpix.o -lcfitsio -lchealpix -L$HEALPIX/lib

mv deg2healpix.* bin/
