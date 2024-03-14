#!/bin/sh
# Compile
dir0=`pwd`
cd ../src
make clean
rm -f ../bin/Filelist2msacs ../bin/PCC_fullpair_1b ../bin/PCC_fullpair_1b_cuda
make Filelist2msacs PCC_fullpair_1b
make PCC_fullpair_1b_cuda
make install
cd $dir0

# Example 1: Correlate a set of sac files from the two stations.
# The data used is from the GEOSCOPE network and was obtained from the Incorporated
# Research Institutions of Seismology (IRIS).
# Pre-processing consists in removing the mean and trends, correcting the instrument
# response to ground velocity, applying a band-pass filter from 4 to 32 mHz and decimating
# to a sampling period of 4 s. Seismograms having much higher energy than average has
# been rejected.

echo -e "\nExample 1:"
#  1) Create a text file (filelist.txt) with the list of sac files for each station.
ls -1 G/CAN/*.LHZ.*.SACvelbp > G/CAN/filelist.txt
ls -1 G/ECH/*.LHZ.*.SACvelbp > G/ECH/filelist.txt
#  2) Correlate them using PCC power of 2 (v=2) from the lag time of -12000 s to 12000 s.
echo "PCC2"
time ../bin/PCC_fullpair_1b G/CAN/filelist.txt G/ECH/filelist.txt pcc tl1=-12000 tl2=12000
#  3) Correlate them using 1-bit correlation from the lag time of -12000 s to 12000 s.
echo "1-bit correlation"
time ../bin/PCC_fullpair_1b G/CAN/filelist.txt G/ECH/filelist.txt cc1b tl1=-12000 tl2=12000
#  4) Correlate them using PCC power of 1 (v=1) from the lag time of -12000 s to 12000 s.
echo "PCC1"
time ../bin/PCC_fullpair_1b G/CAN/filelist.txt G/ECH/filelist.txt pcc v=1 tl1=-12000 tl2=12000
#  5) The same as in 3) but using the GPU
echo "PCC1 using CUDA"
time ../bin/PCC_fullpair_1b_cuda G/CAN/filelist.txt G/ECH/filelist.txt pcc v=1 tl1=-12000 tl2=12000
#  6) Correlate them using WPCC2 from 25 to 330 s periods from the lag time of -12000 s to 12000 s.
time ../bin/PCC_fullpair_1b G/CAN/filelist.txt G/ECH/filelist.txt wpcc2 tl1=-12000 tl2=12000 pmin=25 pmax=330

# Example 2: Gather the sac files of each station in a binary file and correlated them.
echo -e "\nExample 2:"
#  1) Use the Filelist2msacs code to store all the seismograms listed in filelist.txt to 
#     a single binary file.
../bin/Filelist2msacs G/CAN/filelist.txt G/CAN/G.CAN.LHZ.msacs
../bin/Filelist2msacs G/ECH/filelist.txt G/ECH/G.ECH.LHZ.msacs
#  2) Correlate them using PCC power of 1 (v=1) from the lag time of -12000 s to 12000 s.
echo "PCC1 using CUDA reading the data from msacs files"
time ../bin/PCC_fullpair_1b_cuda G/CAN/G.CAN.LHZ.msacs G/ECH/G.ECH.LHZ.msacs imsacs pcc v=1 tl1=-12000 tl2=12000
#  3) Same as 2) but saving the results into a single binary file that can be read with the
#     code of the time-scale phase-weighted stack, https://github.com/sergiventosa/ts-PWS
echo "PCC1 using CUDA reading the data from the msacs files and saving the results into a single binary file"
time ../bin/PCC_fullpair_1b_cuda G/CAN/G.CAN.LHZ.msacs G/ECH/G.ECH.LHZ.msacs imsacs obin pcc v=1 tl1=-12000 tl2=12000
