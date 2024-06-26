Fast phase cross-correlation
============================

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

Software to compute interstation correlations, including fast implementations of the 
phase cross-correlation (Ventosa et al., SRL 2019) and wavelet phase cross-correlation
(Ventosa & Schimmel, 2023) with and without using the GPU.

The software packages of fast phase cross-correlation and [ts-PWS](https://github.com/sergiventosa/ts-PWS) 
stacking are basic building blocks in the design of efficient signal extraction methods 
from interstation correlations.

Main features
-------------
Computes 4 types of correlations:
 * The Standard (geometrically) normalized cross-correlations (GNCC).
 * The 1-bit amplitude normalization followed by the GNCC (1-bit GNCC).
 * The phase cross-correlation (PCC).
 * The wavelet phase cross-correlation (WPCC).

The computations of PCC and WPCC are speed-up in several ways:
 * Both are parallelized in the CPU using OpenMP and in the GPU using CUDA (two independent codes).
 * The computational cost of PCC with power of 2 is reduced to about twice the one of 1-bit GNCC.

Compilation
-----------
To compile execute "make" in the src directory. Use "make clean" to remove 
any previouly compiled code.

 * The [Seismic Analysis Code](http://ds.iris.edu/ds/nodes/dmc/software/downloads/sac/) (SAC) is used to read and write sac files.
 * The FFTW double and single precision libraries are used. If you have to compile them 
   for your system, follow the [fftw.org instructions](http://www.fftw.org/fftw3_doc/Installation-and-Customization.html#Installation-and-Customization). 
 * The SACHOME enviorment variable should provide the path to the directory where sac is
   installed. For example, this can be defined in bash as:  
   export SACHOME=/opt/sac  
   Change "/opt/sac" to your current sac directory if necessary.
 * OpenMP is used to speed up computations. When OpenMP is not available, use 
   make -f makefile_NoOpenMP".

Compile SAC
-----------
The precompiled sac libraries may not work in some systems/compilers. If you use the gcc 
compiler on, e.g., Ubuntu you may have errors similar to:
>  /usr/bin/ld: /opt/sac/lib/sacio.a(getfhv.o): relocation R_X86_64_32 against undefined symbol 'kmlhf' can not be used when making a PIE object; recompile with -fPIC

This can be solved by compiling the source version of SAC. From the source directory of sac do:
>  ./configure --enable-optim=2 --prefix=/opt/sac CFLAGS='-march=native -fPIC' \
>  make \
>  make install

The key detail here is the -fPIC flag, you can adapt the other options to your needs.
The flag --prefix=/opt/sac sets the directory where sac will be installed when doing 
"make install". You can change this directory, e.g., to keep using your current 
version of SAC for other purposes.

Warming up
----------
 1. Read ./examples/example.sh
 2. Execute it, e.g., bash example.sh
 3. Do PCC_fullpair_1b for the parameters usage.
   
Origin of Phase Cross-Correlation
---------------------------------
Schimmel, M., 1999. Phase cross-correlations: Design, comparisons, and applications,
Bulletin of the Seismological Society of America, 89(5), 1366-1378.

Schimmel, M. and Stutzmann, E. & J. Gallart, 2011. Using instantaneous phase coherence 
for signal extraction from ambient noise data at a local to a global scale, Geophysical 
Journal International, 184(1), 494-506, doi:[10.1111/j.1365-246X.2010.04861.x](https://doi.org/10.1111/j.1365-246X.2010.04861.x)
   
Paper to be cited
-----------------
Ventosa S. & M. Schimmel, 2023. Broadband empirical Green’s function extraction
with data adaptive phase correlations, IEEE Transactions on Geoscience and Remote Sensing,
61:1-17, doi:[10.1109/TGRS.2023.3294302](https://doi.org/10.1109/TGRS.2023.3294302)

Ventosa S., Schimmel M. & E. Stutzmann, 2019. Towards the processing of large data 
volumes with phase cross-correlation, Seismological Research Letters, 90(4):1663-1669, 
doi:[10.1785/022019002](https://doi.org/10.1785/022019002)

2024/06/26 Sergi Ventosa Rahuet (sergiventosa(at)hotmail.com)
