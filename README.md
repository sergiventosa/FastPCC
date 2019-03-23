Fast Phase Cross-Correlation
============================

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

Software to compute interstation correlations of seismic ambient noise, including fast implementations of the 
standard, 1-bit and phase cross-correlations. (Ventosa et al., SRL 2019).

Main features
-------------
Computes 3 types of correlations:
 * The standard (geometrically-) normalized cross-correlations (GNCC).
 * The 1-bit amplitude normalization followed by the GNCC (1-bit GNCC).
 * The phase cross-correlation (PCC).

The computations of PCC are speed-up in several ways:
 * PCC is parallelized in the CPU using OpenMP and in the GPU (optional) using CUDA.
 * The computational cost of PCC with power of 2 is reduced to about twice the one of 1-bit GNCC.

Compilation
-----------
To compile execute "make" in the src directory. Use "make clean" to remove 
any previouly compiled code.

 * The Seismic Analysis Code (SAC) is used to read and write sac files.
 * The FFTW double and single precision libraries.
 * The SACHOME enviorment variable should provide the path to directory where sac is
   installed. For example, in bash this can be defined as:  
   export SACHOME=/opt/sac  
   Change "/opt/sac" to your current sac directory if necessary.
 * OpenMP is used to speed up computations. When OpenMP is not available, use 
   make -f makefile_NoOpenMP
 * When a Nvidia GPU is available, we use CUDA to speed up PCC with power of 1. 
   Do "make PCC_fullpair_1a_cuda" or "make -f makefile_NoOpenMP PCC_fullpair_1a_cuda".
   
Origin of Phase Cross-Correlation
---------------------------------
Schimmel, M., 1999. Phase cross-correlations: Design, comparisons, and applications,
Bulletin of the Seismological Society of America, 89(5), 1366-1378.

Schimmel, M. and Stutzmann, E. & J. Gallart, 2011. Using instantaneous phase coherence 
for signal extraction from ambient noise data at a local to a global scale, Geophysical 
Journal International, 184(1), 494-506, doi:10.1111/j.1365-246X.2010.04861.x
   
Paper to be cited
-----------------
Ventosa S., Schimmel M. & E. Stutzmann, 2019. Towards the processing of large data 
volumes with phase cross-correlation, Seismological Research Letters.

2019/03/21 Sergi Ventosa Rahuet (sergiventosa(at)hotmail.com)
