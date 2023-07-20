/*****************************************************************************/
/* Software to compute interstation correlations, including fast implementa- */
/* tions of geometrically-normalized (CCGN), 1-bit correlation (1-bit CCGN), */
/* and phase cross-correlations (PCC) with and without using the GPU.        */
/*                                                                           */
/* Authors: Sergi Ventosa Rahuet (sergiventosa@hotmail.com)                  */
/*                                                                           */
/* Main features:                                                            */
/*   - Compute CCGN, 1-bit CCGN, PCC & WPCC.                                 */
/*                                                                           */
/* **** 2016 ****                                                            */
/* May19 (1a) Fast implementation of PCC with power of 2 using fftw3         */
/*  - Not taking care of zero samples.                                       */
/* **** 2018 ****                                                            */
/* Mar21 (1b) A few features has been added along the way.                   */
/*  - Compute PCC with any power.                                            */
/*  - Compute CCGN using fftw3.                                              */
/*  - Clipping.                                                              */
/*****************************************************************************/
/* Temporal: Save loc '' of Geoscope (G) stations as '00'                    */ 
/*****************************************************************************/
/* **** 2020 ****                                                            */
/* Jun25 (1b) A few minor updates.                                           */
/*  - Consider milli-seconds in MakePairedLists() and SortTraces()           */
/*  - Relax criteria to accept SAC files in ReadManySacs(). The default      */
/*    sequence length (first trace) can be modified with the Nmax parameter. */
/*    Too long sequences are now cut instead of rejected.                    */
/* Sep16 (1b) A few minor updates.                                           */
/*   - The O sac header field now contains the absolute time difference of   */
/*     the begin times of the traces (the actual "zero" lag time). Pairs     */
/*     misaligned by more than half a second are still rejected (alignment   */
/*     must be done in advance).                                             */
/*   - Modified the format of the output binary file (msacs) to save the     */
/*     actual "zero" lag time for each correlation pair.                     */
/* **** 2021 ****                                                            */
/* Jan21 (1b) Bug correction                                                 */
/*   - Station 1 is the virtual source and station 2 the virtual receiver    */
/*     (event and station in the sac header, respectively), but computations */
/*     were done using the opposite criterion.                               */
/* Abr14 (1d) Minor update to compile with SAC v102.0                        */
/* **** 2023 ****                                                            */
/* Jan05 (1d)                                                                */
/*   - Release the code to compute the wavelet phase cross-correlation       */
/*     (WPCC).                                                               */
/*   - Bug correction:                                                       */
/*       Correlations were not computed if nl1>0 or tl1>0.                   */
/*       Station latitud and longitud are not required any more.             */
/*****************************************************************************/

#include <complex.h>  /* fftw3.h use C99 complex types when complex.h is included before. */
#include <fftw3.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <sacio.h>
#include <float.h>
#include "FFTapps.h"
#include "ReadManySacs.h"
#include "rotlib.h"
#include "sac2bin.h"
#include "sph.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

typedef struct {
	double        tl1;
	double        tl2;
	double        v;
	int           nl1;
	int           nl2;
	unsigned int  iformat;
	unsigned int  oformat;
	unsigned int  pcc;
	unsigned int  wpcc;
	unsigned int  ccgn;
	unsigned int  cc1b;
	unsigned int  clip;
	int           mincc;
	int           Nmax;     /* Maximum number of samples per trace, default defined by the first file. */
	double        std;
	double        mindist;  /* Min distance in degrees. */ 
	double        maxdist;  /* Max distance in degrees. */ 
	double        pmin;     /* wpcc2: Shorter period. */
	double        pmax;     /* wpcc2: Longest period. */
	double        VR;       /* wpcc2: Velocity setting the longest period having 3 wavelength. */
	unsigned int  V;        /* wpcc2: Number of voices (default 2). */
	int           type;     /* wpcc2: Wavelet family (default MexHat). */
	double        op1;      /* wpcc2: Center frequency of the mother wavelet, Only for the Morlet wavelet. */
	double        awhite[2];
	char          *fin1;
	char          *fin2;
	char          *obinprefix;
	int           autopair; /* 0: Pair filelists line per line, 1: pair filelists automatically according to the metadata (default 1). */
	int           acc;      /* 0: cross-correlation, 1: autocorrelation */
	int           verbose;  /* 0: Silent mode (no message), 1: some message (default), 2: a few more. */
} t_PCCmatrix;

typedef struct {
	unsigned int  ind;
	time_t        time;
	int32_t       msec;
} t_elem;

int PCCfullpair_main (t_PCCmatrix *fpcc);

int StoreInManySacs (float **y, unsigned int L, unsigned int Tr, int Lag1, t_HeaderInfo *SacHeader1, 
	t_HeaderInfo *SacHeader2, float dt, char *ccname, int verbose);
int StoreInManyBins (float **y, unsigned int L, unsigned int Tr, int Lag1,  t_HeaderInfo *SacHeader1, 
	t_HeaderInfo *SacHeader2, float dt, char *ccname, char *prefix, int verbose);
int wrsac(char *filename, char *kstnm, float beg, float dt, float *y, int nsamp, t_HeaderInfo *ptr1, t_HeaderInfo *ptr2);

void infooo();
void usage();
void clipping (float *x, unsigned int N);
int RemoveOutlierTraces (float **xOut[], t_HeaderInfo *SacHeader[], unsigned int *Tr0, float *std, float nstd);
int SortTraces (float **xOut[], t_HeaderInfo *SacHeader[], unsigned int *Tr0);
int MakePairedLists (float **xOut1[], t_HeaderInfo *SacHeader1[], unsigned int *TrOut1, 
	float **xOut2[], t_HeaderInfo *SacHeader2[], unsigned int *TrOut2);
int CheckPairs (t_HeaderInfo *SacHeader1, unsigned int Tr1, t_HeaderInfo *SacHeader2, unsigned int Tr2);
int AveWhite (float **x0, unsigned int N, unsigned int Tr, double freq[2], double dt);
float qmedian(float a[], int n);
float qabsmedian(float input[], int n);

int RDint (int * const x, const char *str);
int RDuint (unsigned int * const x, const char *str);
int RDdouble (double * const x, const char *str);
int RDdouble_array (double * const x, char * const str, unsigned int N);

/* Main function: Mainly reading parameters. */
int main(int argc, char *argv[]) {
	t_PCCmatrix fpcc = {0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -3, 0, {0, 0}, NULL, NULL, NULL, 1, 0, 1};  /* Default parameters. */
	int i, er = 0;
	
	if (argc < 3) {
		if (argc == 2 && !strncmp(argv[1], "info", 4)) infooo();
		else usage();
		return 0;
	}
	
	fpcc.fin1 = argv[1];
	fpcc.fin2 = argv[2];
	
	fpcc.acc = (strcmp(fpcc.fin1, fpcc.fin2) == 0) ? 1 : 0;
	
	for (i=1; i<argc; i++) {
		if (!strncmp(argv[i], "tl1=", 4))      er += RDdouble(&fpcc.tl1, argv[i] + 4);
		else if (!strncmp(argv[i], "tl2=",   4)) er += RDdouble(&fpcc.tl2, argv[i] + 4);
		else if (!strncmp(argv[i], "v=",     2)) er += RDdouble(&fpcc.v,   argv[i] + 2);
		else if (!strncmp(argv[i], "nl1=",   4)) er += RDint(&fpcc.nl1,   argv[i] + 4);
		else if (!strncmp(argv[i], "nl2=",   4)) er += RDint(&fpcc.nl2,   argv[i] + 4);
		else if (!strncmp(argv[i], "isac",   4)) fpcc.iformat = 1;
		else if (!strncmp(argv[i], "imsacs", 6)) fpcc.iformat = 2;
		else if (!strncmp(argv[i], "osac",   4)) fpcc.oformat = 1;
		else if (!strncmp(argv[i], "obin",   4)) {
			fpcc.oformat = 2;
			if (!strncmp(argv[i], "obin=",  5)) fpcc.obinprefix = argv[i] + 5;
		} 
		else if (!strncmp(argv[i], "pcc",    3)) fpcc.pcc  = 1;
		else if (!strncmp(argv[i], "wpcc",   4)) fpcc.wpcc = 1;
		else if (!strncmp(argv[i], "ccgn",   4)) fpcc.ccgn = 1;
		else if (!strncmp(argv[i], "cc1b",   4)) fpcc.cc1b = 1;
		else if (!strncmp(argv[i], "clip",   4)) fpcc.clip = 1;
		else if (!strncmp(argv[i], "std=",   4)) er += RDdouble(&fpcc.std, argv[i] + 4);
		else if (!strncmp(argv[i], "mindist=", 8)) er += RDdouble(&fpcc.mindist, argv[i] + 8);
		else if (!strncmp(argv[i], "maxdist=", 8)) er += RDdouble(&fpcc.maxdist, argv[i] + 8);
		else if (!strncmp(argv[i], "pmin=",  5)) er += RDdouble(&fpcc.pmin, argv[i] + 5);
		else if (!strncmp(argv[i], "pmax=",  5)) er += RDdouble(&fpcc.pmax, argv[i] + 5);
		else if (!strncmp(argv[i], "VR=",    3)) er += RDdouble(&fpcc.VR, argv[i] + 3);
		else if (!strncmp(argv[i], "mincc=", 6)) er += RDint(&fpcc.mincc, argv[i] + 6);
		else if (!strncmp(argv[i], "Nmax=",  5)) er += RDint(&fpcc.Nmax, argv[i]  + 5);
		else if (!strncmp(argv[i], "V=",     2)) er += RDuint(&fpcc.V, argv[i] + 2);
		else if (!strncmp(argv[i], "type=",  5)) er += RDint(&fpcc.type, argv[i] + 5);
		else if (!strncmp(argv[i], "w0=",    3)) er += RDdouble(&fpcc.op1, argv[i] + 3);
		else if (!strncmp(argv[i], "awhite=",7)) er += RDdouble_array(fpcc.awhite, argv[i] + 7, 2);
		else if (!strncmp(argv[i], "NoAutoPairing", 13)) fpcc.autopair = 0;
		else if (!strncmp(argv[i], "verbose=", 8)) er += RDint(&fpcc.verbose, argv[i] + 8);
		else if (!strncmp(argv[i], "info",   4)) {
			infooo();
			return 0;
		}
	}
	
	if ( !fpcc.autopair && fpcc.std > 0) {
		printf("PCCfullpair: Warning, the std option cannot be used without automatic trace pairing.\n");
		fpcc.std = 0;
	}
	er = PCCfullpair_main(&fpcc); /* The one who make the job. */
	return er;
}

int PCCfullpair_main (t_PCCmatrix *fpcc) {
	t_HeaderInfo *SacHeader1=NULL, *SacHeader2=NULL;
	float *std=NULL;
	float dt, dt1;
	double pmin, pmax;
	float **x1=NULL, **x2=NULL, **y=NULL, *px;
	double lat1=-90, lon1=0, lat2=90, lon2=0, gcarc, da2;
	unsigned int tr, Tr, Tr1, Tr2, n, N, N1;
	int Lag1, Lag2, ia1, L, nerr=0, nerr1=0, stloc=1;
	char nickpcc[16]; /* Up to the first 8 are saved in the sac header. */
	
	/* Input checkings */
	if (fpcc == NULL) {       printf("PCCfullpair_main: NULL input\n");           return -1; }
	if (fpcc->fin1 == NULL) { printf("PCCfullpair_main: NULL input filename1\n"); return -1; }
	if (fpcc->fin2 == NULL) { printf("PCCfullpair_main: NULL input filename2\n"); return -1; }
	
	/* Read input files */
	if (fpcc->iformat == 1) {
		if ( -4 == (nerr1 = ReadLocation (&lat1, &lon1, fpcc->fin1)) ) {
			printf("PCCfullpair_main: Warning, the station location of %s is not available, following with 0.\n", fpcc->fin1);
			stloc = 0;
		}
		if (fpcc->acc == 0) {
			if ( -4 == (nerr1 = ReadLocation (&lat2, &lon2, fpcc->fin2)) ) {
				printf("PCCfullpair_main: Warning, the station location of %s is not available, following with 0.\n", fpcc->fin2);
				stloc = 0;
			}
		}
	} else if (fpcc->iformat == 2) { 
		if ( 0 != ReadLocation_ManySacsFile (&lat1, &lon1, fpcc->fin1) ) stloc = 0;
		if (fpcc->acc == 0) {
			if ( 0 != ReadLocation_ManySacsFile (&lat2, &lon2, fpcc->fin2) ) stloc = 0;
		}
	} else { 
		printf("PCCfullpair_main: Unknown format."); 
		nerr = 5; 
	}
	
	if ( fpcc->acc == 1) {
		lat2 = lat1; 
		lon2 = lon1;
	}
	
	if (!stloc) {
		if (fpcc->mindist > 0) {
			fpcc->mindist = 0;
			printf("PCCfullpair_main: Warning, the station location is not available, Following withou mindist.\n");
		}
		if (fpcc->maxdist > 0) {
			fpcc->maxdist = 0;
			printf("PCCfullpair_main: Warning, the station location is not available, Following without maxdist.\n");
		}
		if (fpcc->VR > 0) {
			fpcc->VR = 0;
			printf("PCCfullpair_main: Warning, the station location is not available, Following without VR.\n");
		}
		gcarc = 0;
	} else {
		lat1 *= DEG2RAD;
		lon1 *= DEG2RAD;
		lat2 *= DEG2RAD;
		lon2 *= DEG2RAD;
		sph_gcarc (&gcarc, lat1, lon1, &lat2, &lon2, 1);
		gcarc *= RAD2DEG;
		
		if (fpcc->mindist > 0 || fpcc->maxdist > 0) { /* Check minimum distance */
			if (fpcc->mindist > 0 && gcarc < fpcc->mindist) {
				printf("PCCfullpair_main: The interstation distance is smaller than the %f degrees requiered on %s - %s.\n", fpcc->mindist, fpcc->fin1, fpcc->fin2);
				return 0;
			}
		
			if (fpcc->maxdist > 0 && gcarc > fpcc->maxdist) {
				printf("PCCfullpair_main: The interstation distance is higher than the %f degrees requiered on %s - %s.\n", fpcc->maxdist, fpcc->fin1, fpcc->fin2);
				return 0;
			}
		}
	}
	
	N1 = fpcc->Nmax;
	if (fpcc->iformat == 1)
		nerr1 = ReadManySacs (&x1, &SacHeader1, NULL, &Tr1, &N1, &dt1, fpcc->fin1);
	else if (fpcc->iformat == 2)
		nerr1 = Read_ManySacsFile (&x1, &SacHeader1, &Tr1, &N1, &dt1, fpcc->fin1);
	
	if (Tr1==0) { printf("PCCfullpair_main: %s is empty.", fpcc->fin1); return 0; }
	if (Tr1 < fpcc->mincc) {
		printf("PCCfullpair_main: Too few sequences from station %s (%d < %d)\n", fpcc->fin1, Tr1, fpcc->mincc); 
		Destroy_FloatArrayList (x1, Tr1);
		free (SacHeader1);
		return 0;
	}
	if (nerr1) { printf("PCCfullpair_main: Something went wrong when reading the data from station 1! (nerr = %d)\n", nerr1); return nerr1; }
	
	nerr = RemoveZeroTraces (&x1, &SacHeader1, &Tr1, N1);
	if (nerr) printf("PCCfullpair_main: Something went wrong when RemoveZeroTraces of station 1! (nerr = %d)\n", nerr);
	
	N = fpcc->Nmax;
	if (fpcc->acc == 0) {
		if (fpcc->iformat == 1)
			nerr  = ReadManySacs (&x2, &SacHeader2, NULL, &Tr2, &N, &dt, fpcc->fin2);
		else if (fpcc->iformat == 2)
			nerr  = Read_ManySacsFile (&x2, &SacHeader2, &Tr2, &N, &dt, fpcc->fin2);
		else { 
			printf("PCCfullpair_main: Unknown format."); 
			nerr = 5; 
		}
		
		if (Tr2==0) { printf("PCCfullpair_main: %s is empty.", fpcc->fin2); return 0; }
		if (Tr2 < fpcc->mincc) {
			printf("PCCfullpair_main: Too few sequences from station %s (%d < %d)\n", fpcc->fin2, Tr2, fpcc->mincc); 
			Destroy_FloatArrayList (x1, Tr1);
			Destroy_FloatArrayList (x2, Tr2);
			free (SacHeader1);
			free (SacHeader2);
			return 0;
		}
		if (nerr)  { printf("PCCfullpair_main: Something went wrong when reading the data from station 2! (nerr = %d)\n", nerr);  return nerr; }
		
		nerr = RemoveZeroTraces (&x2, &SacHeader2, &Tr2, N);
		if (nerr) printf("PCCfullpair_main: Something went wrong when RemoveZeroTraces of station 2! (nerr = %d)\n", nerr);
		
		if (N1 != N) printf("PCCfullpair_main: These traces have different lengths: %u:%u\n", N1, N);
		if (dt1 != dt) printf("PCCfullpair_main: These stations have different samplings: %11.9f:%11.9f\n", dt1, dt);
	} else {
		x2 = x1;
		SacHeader2 = SacHeader1;
		Tr2 = Tr1;
		N   = N1;
		dt  = dt1;
	}
	
	/* Outliers and clippling for station 1 */
	/* Data std. (The signal should have no mean). */
	if (fpcc->std) {
		if (NULL == (std = (float *)calloc(Tr1, sizeof(float)) )) {
			Destroy_FloatArrayList(x1, Tr1);
			free(SacHeader1);
			return 4;
		}
	
		for (tr=0; tr<Tr1; tr++) {
			px = x1[tr];
			da2 = 0;
			for (n=0; n<N; n++) da2 += px[n] * px[n];
			std[tr] = (float)sqrt(da2/N);
		}
	}

	/* Clipping */
	if (fpcc->clip)
		for (tr=0; tr<Tr1; tr++) clipping (x1[tr], N);
	
	/* Remove traces having much higher or lower energy than the others ones. */
	if (fpcc->std) {
		nerr = RemoveOutlierTraces (&x1, &SacHeader1, &Tr1, std, fpcc->std);
		if (nerr) { printf("PCCfullpair_main: Something went wrong when RemoveOutlierTraces of station 1! (nerr = %d)\n", nerr); return nerr; }
	}
	
	/* Outliers and clippling for station 2 */
	if (fpcc->acc == 0) {
		if (fpcc->std) {
			if (NULL == (std = (float *)calloc(Tr2, sizeof(float)) )) {
				Destroy_FloatArrayList(x2, Tr2);
				free(SacHeader2);
				return 4;
			}
		
			for (tr=0; tr<Tr2; tr++) {
				px = x2[tr];
				da2 = 0;
				for (n=0; n<N; n++) da2 += px[n] * px[n];
				std[tr] = (float)sqrt(da2/N);
			}
		}
	
		/* Clipping */
		if (fpcc->clip)
			for (tr=0; tr<Tr2; tr++) clipping (x2[tr], N);
		
		/* Remove traces having much higher or lower energy than the others ones. */
		if (fpcc->std) {
			nerr = RemoveOutlierTraces (&x2, &SacHeader2, &Tr2, std, fpcc->std);
			if (nerr) { printf("PCCfullpair_main: Something went wrong when RemoveOutlierTraces of station 2! (nerr = %d)\n", nerr); return nerr; }
			free(std);
		}
	} else Tr2 = Tr1;
	
	if (fpcc->autopair) {
		/* Sort out each list. */
		SortTraces (&x1, &SacHeader1, &Tr1);
		if (fpcc->acc == 0) SortTraces (&x2, &SacHeader2, &Tr2);
	
		/* Remove time sequences not present on both stations. */
		if (fpcc->acc == 0) MakePairedLists (&x1, &SacHeader1, &Tr1, &x2, &SacHeader2, &Tr2);
	} else {
		nerr = CheckPairs(SacHeader1, Tr1, SacHeader2, Tr2);
		if (nerr) { printf("PCCfullpair_main: Something went wrong when checking the pairs! (nerr = %d)\n", nerr); return nerr; }
	}
	Tr = Tr1;
	
	/* Whittenings */
	if (fpcc->awhite[0] > 0 && fpcc->awhite[1] > fpcc->awhite[0]) {
		AveWhite (x1, N, Tr, fpcc->awhite, dt);
		AveWhite (x2, N, Tr, fpcc->awhite, dt);
	}
	
	/* Calculate lags. */
	if (fpcc->nl1) Lag1 = fpcc->nl1;
	else if (fpcc->tl1) Lag1 = (int)round(fpcc->tl1 / (double)dt);
	else Lag1 = 0;
	if (fpcc->nl2) Lag2 = fpcc->nl2;
	else if (fpcc->tl2) Lag2 = (int)round(fpcc->tl2 / (double)dt);
	else Lag2 = 0;
	
	if (Lag1 > Lag2) { ia1 = Lag1; Lag1 = Lag2; Lag2 = ia1; }
	if (abs(Lag1) >= N || abs(Lag2) >= N) { 
		printf("PCCfullpair_main: TOO LARGE LAGS!!! The modulus of the Lags have to be lower than the sequence length.\n");
		return 6; 
	}
	L = Lag2-Lag1+1;
	
	printf("Lag1 = %d, Lag2 = %d, L = %d, N = %d, Tr = %d, gcarc = %f\n", Lag1, Lag2, L, N, Tr, gcarc);
	if (Tr <= fpcc->mincc) {
		if (!Tr) printf("NO INTERSTATION CORRELATION TO BE COMPUTED.\n");
		else printf("ONLY %d INTERSTATION CORRELATION COULD BE COMPUTED.\n", Tr); 
	} else {
		/* Calculate pmin and pmax parameters required in wpcc2 */
		pmin = 1.25 * sqrt(2)*PI;
		pmax = pmin * pow(2, 3.5);
		if (fpcc->wpcc) {  /* Wavelet PCCs: */
			if (fpcc->pmin != 0 && fpcc->pmax != 0) {
				pmin = fpcc->pmin / dt;
				if (fpcc->VR == 0) 
					pmax = fpcc->pmax / dt;
				else {
					pmax = 111.19*gcarc/(3*fpcc->VR);
					if (pmax > fpcc->pmax) pmax = fpcc->pmax;
					pmax /= dt;
				}
			}
			printf("pmin = %f, pmax = %f\n", pmin, pmax);
		}
		
		if (fpcc->v==2)      strcpy(nickpcc, "pcc2");
		else if (fpcc->v==1) strcpy(nickpcc, "pcc1");
		else sprintf(nickpcc, "pcc%.1f", fpcc->v);
		
		/* Output array memory */
		if (NULL == (y = Create_FloatArrayList (L, Tr) )) {
			printf ("PCCfullpair_main: Out of memory on the output array (%d x %d)\n", L, Tr);
			nerr = 4;
		} else {
			/* The actual cross-correlations */
			CorrectRevesedPolarity (x1, N, Tr, SacHeader1); /* Corrects for sign-flips on a component. */
			if (fpcc->acc == 0) CorrectRevesedPolarity (x2, N, Tr, SacHeader2);
			if (fpcc->pcc) {  /* PCCs: */
				if (fpcc->v==2)      pcc2_set (y, x1, x2, N, Tr, Lag1, Lag2);
				else if (fpcc->v==1) pcc1_set (y, x1, x2, N, Tr, Lag1, Lag2);
				else pcc_set (y, x1, x2, N, Tr, fpcc->v, Lag1, Lag2);
				if (fpcc->oformat==1) 
					StoreInManySacs (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, nickpcc, fpcc->verbose);
				else if (fpcc->oformat==2) 
					StoreInManyBins (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, nickpcc, fpcc->obinprefix, fpcc->verbose);
			}
			
			if (fpcc->wpcc) {  /* Wavelet PCCs: */
				tspcc2_set (y, x1, x2, N, Tr, Lag1, Lag2, pmin, pmax, fpcc->V, fpcc->type, fpcc->op1);
				if (fpcc->oformat==1) 
					StoreInManySacs (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, "wpcc2", fpcc->verbose);
				else if (fpcc->oformat==2) 
					StoreInManyBins (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, "wpcc2", fpcc->obinprefix, fpcc->verbose); 
			}			

			if (fpcc->ccgn) {  /* GNCCs */
				ccgn_set (y, x1, x2, N, Tr, Lag1, Lag2);
				if (fpcc->oformat==1) 
					StoreInManySacs (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, "ccgn", fpcc->verbose);
				else if (fpcc->oformat==2) 
					StoreInManyBins (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, "ccgn", fpcc->obinprefix, fpcc->verbose);
			}
			
			if (fpcc->cc1b) {  /* 1-bit + GNCCs */
				cc1b_set (y, x1, x2, N, Tr, Lag1, Lag2);
				if (fpcc->oformat==1) 
					StoreInManySacs (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, "cc1b", fpcc->verbose);
				else if (fpcc->oformat==2) 
					StoreInManyBins (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, "cc1b", fpcc->obinprefix, fpcc->verbose);
			}
			
			Destroy_FloatArrayList (y, Tr);
		}
	}

	Destroy_FloatArrayList (x1, Tr);
	free (SacHeader1);
	if (fpcc->acc == 0) {
		Destroy_FloatArrayList (x2, Tr);
		free (SacHeader2);
	}
	return 0;
}

void infooo() {
	puts("\nThis program computes the geometrically-normalized (CCGN), 1-bit correlation (1-bit CCGN), phase cross-correlations (PCC) and wavelet phase cross-correlation (WPCC) between seismograms from two stations.");
	puts("I developed this program for Ventosa et al. (2017) and I further developed and presented it in Ventosa et al. (2019) and Ventosa & Schimmel (2003). Information on the PCC is published in Schimmel (1999).\n");
	puts("Schimmel, M., 1999.  Phase cross-correlations: Design, comparisons, and applications, Bulletin of the Seismological Society of America, 89(5), 1366-1378.\n");
	puts("Ventosa, S., Schimmel, M., & E. Stutzmann, 2017. Extracting surface waves, hum and normal modes: Time-scale phase-weighted stack and beyond, Geophysical Journal International, 211(1), 30-44, doi:10.1093/gji/ggx284.\n");
	puts("Ventosa, S., Schimmel, M., & E. Stutzmann, 2019. Towards the processing of large data volumes with phase cross-correlation, Seismological Research Letters, 90(4), 1663-1669, doi:10.1785/0220190022.\n"); 
	puts("Ventosa, S. & M. Schimmel, 2023. Broadband empirical Green’s function extraction with data adaptive phase correlations, IEEE Transactions on Geoscience and Remote Sensing, doi:10.1109/TGRS.2023.3294302.\n");
	puts("AUTHOR: Sergi Ventosa Rahuet (sergiventosa(at)hotmail.com)");
	puts("Last modification: 13/07/2023\n");
}

void usage() {
	puts("\nCompute interstation correlations, including fast implementations of geometrically-normalized (CCGN),"); 
	puts("1-bit correlation (1-bit CCGN), phase cross-correlations (PCC) and wavelet phase cross-correlation (WPCC)");
	puts("with and without using the GPU.");
	puts("");
	puts("USAGE: PCC_fullpair_1b filelist1 filelist2 parameters");
	puts("  filelist1:  text file containing a list of SAC files for station 1. One filename per line.");
	puts("  filelist2:  idem to filelist1 but for station 2.");
	puts("  parameters: in arbitrary order without any blank around '='.");
	puts("");
	puts("The traces are paired automatically according to their date and time header information (nzxxx header ");
	puts("variables) using each trace only in one pair. All traces must have the same begin time (b), number of");
	puts("samples (nsmpl) and sampling interval.");
	puts("");
	puts("Most commonly used parameters");
	puts("  nl1=   : starting sample lag. nl1=0 by default.");
	puts("  nl2=   : ending sample lag. nl2=0 by default.");
	puts("  tl1=   : starting relative time lag in seconds. Modifies nl1. tl1=0 by default.");
	puts("  tl2=   : ending relative time lag in seconds. Modifies nl2. tl2=0 by default.");
	puts("  Nmax=  : maximum number of samples per sequence. The default is the length of the first"); 
	puts("           sequence in the filelist.");
	puts("  ccgn   : compute geometrically normalized cross-correlation. Not computed by default.");
	puts("  cc1b   : compute 1-bit amplitude normalization followed by ccgn. Not computed by default.");
	puts("  pcc    : compute phase cross-correlation. Not computed by default.");
	puts("  wpcc   : compute wavelet phase cross-correlation. Not computed by default.");
	puts("  v      : pcc power, sum(|a+b|^v - |a+b|^v). Default is v=2");
	puts("  verbose: ");
	puts("  info   : write background and main references to screen.");
	puts("           Just type: PCC_fullpair_1b info.");
	puts("");
	puts("Input/Output data format");
	puts("  isac   : Input traces are in the SAC format (default)");
	puts("  imsacs : Input traces are garthered in two files, one per station, replacing filelist1 and filelist2");
	puts("           above. Use the Filelist2msacs code to group the sac files in single msac file.");
	puts("  osac   : Output interstation correlations are saved in many files in the SAC format (default)");
	puts("  obin   : Output interstation correlations are saved in one file to speed up data reading in");
	puts("           the ts-PWS stacking code, https://github.com/sergiventosa/ts-PWS.  ");
	puts("");
	puts("Additional functionalities");
	puts("  clip   : clip input sequences before the correlations at 4*MAD/0.6745, about 4 sigmas.");
	puts("  std=   : remove sequences whose samples have a standard deviation n times higher than average ");
	puts("           standard deviation of all traces.");
	puts("  awhite=f1,f2 : smooth spectral whitening in the frequency band f1 - f2 (f1 < f2) using a"); 
	puts("                 Blackman window of 11 samples.");
	puts("  NoAutoPairing: Disables the automatic trace pairing. The traces are paired line per line.\n");
	puts("");
	puts("EXAMPLES");
	puts("  Computes PCC of power 1 and CCGN between the traces listed in filelist1.txt and filelist2.txt");
	puts("   recorded at the same time (i.e., equal SAC header variables nzxxx):");
	puts("     PCC_fullpair_1b filelist1.txt filelist2.txt tl1=-1000 tl2=1000 ccgn pcc v=1");
	puts("");
	puts("  The traces listed in filelist1.txt and filelist2.txt typically share the same station, channel");
	puts("  and component codes.");
	puts("");
	puts("  PCC of power 2 and 1-bit CCGN between the traces stored in sta1.msacs and sta2.msacs:");
	puts("     Filelist2msacs filelist1.txt sta1.msacs");
	puts("     Filelist2msacs filelist2.txt sta2.msacs");
	puts("     PCC_fullpair_1b sta1.msacs sta2.msacs imsacs tl1=-1000 tl2=1000 cc1b pcc v=2");
	puts("");
	puts("AUTHOR: Sergi Ventosa, 13/07/2023");
	puts("Version 1.1.0");
	puts("Please, do not hesitate to send bugs, comments or improvements to sergiventosa(at)hotmail.com\n");
}

int isstring (char *s, unsigned int N) {
	unsigned int n;
	
	for (n=0; n<N; n++) 
		if (!isalnum(s[n]) && !isspace(s[n]) && !ispunct(s[n]) && s[n] != 0 ) return n+1;
	return 0;
}

int check_header (t_HeaderInfo *h) {
	int n, out = 0;
	
	if (fabs(h->stla) > 90 || fabs(h->stlo) > 180) out = 1;
	if (isstring(h->net, 8) || isstring(h->sta, 8) || isstring(h->chn, 8) || isstring(h->loc, 8) ) out = 1; 
	
	if (out) {
		printf("\nWARNING: Station HeaderInfo corrupted\n");
		printf("  lat=%f\n lon=%f\n", h->stla, h->stlo);
		printf("  net=%.8s\n sta=%.8s\n loc=%.8s\n chn=%.8s\n", h->net, h->sta, h->loc, h->chn);
		if (fabs(h->stla) > 90)  printf("stla is not fine\n");
		if (fabs(h->stlo) > 180) printf("stlo is not fine\n");
		if ((n = isstring(h->net, 8))) { n--; printf("net[%d] is not fine (%d)\n", n, h->net[n]); }
		if ((n = isstring(h->sta, 8))) { n--; printf("sta[%d] is not fine (%d)\n", n, h->sta[n]); }
		if ((n = isstring(h->chn, 8))) { n--; printf("chn[%d] is not fine (%d)\n", n, h->chn[n]); }
		if ((n = isstring(h->loc, 8))) { n--; printf("loc[%d] is not fine (%d)\n", n, h->loc[n]); }
	}
	return out;
}

int check_binheader (t_ccheader *h) {
	int n, out = 0;
	
	if (fabs(h->stlat1) > 90 || fabs(h->stlon1) > 180 || fabs(h->stlat2) > 90 || fabs(h->stlon2) > 180) out = 1;
	if (isstring(h->net1, 8) || isstring(h->sta1, 8) || isstring(h->chn1, 8) || isstring(h->loc1, 8) ) out = 1; 
	if (isstring(h->net2, 8) || isstring(h->sta2, 8) || isstring(h->chn2, 8) || isstring(h->loc2, 8) ) out = 1; 
	
	if (out) {
		printf("\nWARNING: Correlation header corrupted\n");
		printf(" stlat1=%f\n stlon2=%f\n", h->stlat1, h->stlon1);
		printf(" stlat2=%f\n stlon2=%f\n", h->stlat2, h->stlon2);
		printf(" net1=%.8s\n sta1=%.8s\n loc1=%.8s\n chn1=%.8s\n", h->net1, h->sta1, h->loc1, h->chn1);
		printf(" net2=%.8s\n sta2=%.8s\n loc2=%.8s\n chn2=%.8s\n", h->net2, h->sta2, h->loc2, h->chn2);
		if (fabs(h->stlat1) > 90)  printf("stlat1 is not fine\n");
		if (fabs(h->stlon1) > 180) printf("stlon1 is not fine\n");
		if (fabs(h->stlat2) > 90)  printf("stlat2 is not fine\n");
		if (fabs(h->stlon2) > 180) printf("stlon2 is not fine\n");
		if ((n = isstring(h->net1, 8))) { n--; printf("net1[%d] is not fine (%d)\n", n, h->net1[n]); }
		if ((n = isstring(h->sta1, 8))) { n--; printf("sta1[%d] is not fine (%d)\n", n, h->sta1[n]); }
		if ((n = isstring(h->chn1, 8))) { n--; printf("chn1[%d] is not fine (%d)\n", n, h->chn1[n]); }
		if ((n = isstring(h->loc1, 8))) { n--; printf("loc1[%d] is not fine (%d)\n", n, h->loc1[n]); }
		if ((n = isstring(h->net2, 8))) { n--; printf("net2[%d] is not fine (%d)\n", n, h->net2[n]); }
		if ((n = isstring(h->sta2, 8))) { n--; printf("sta2[%d] is not fine (%d)\n", n, h->sta2[n]); }
		if ((n = isstring(h->chn2, 8))) { n--; printf("chn2[%d] is not fine (%d)\n", n, h->chn2[n]); }
		if ((n = isstring(h->loc2, 8))) { n--; printf("loc2[%d] is not fine (%d)\n", n, h->loc2[n]); }
	}
	return out;
}

int StoreInManySacs (float **y, unsigned int L, unsigned int Tr, int Lag1, t_HeaderInfo *SacHeader1, 
		t_HeaderInfo *SacHeader2, float dt, char *ccname, int verbose) {
	char dat[30], outfilename[100], loc1[9], loc2[9];
	t_HeaderInfo *hdr1, *hdr2;
	unsigned int tr1, trOut;
	float *sig;
	
	trOut = 0;
	if (verbose >= 2) {
		check_header (SacHeader1);
		check_header (SacHeader2);
	}
	for (tr1=0; tr1<Tr; tr1++) {
		hdr1 = &SacHeader1[tr1];
		hdr2 = &SacHeader2[tr1];
		strncpy(loc1, hdr1->loc, 8);
		// if ( !strncmp(hdr1->net, "G", 4) && !strcmp(loc1, "") ) strcpy(loc1, "00");
		strncpy(loc2, hdr2->loc, 8);
		// if ( !strncmp(hdr2->net, "G", 4) && !strcmp(loc2, "") ) strcpy(loc2, "00");
		
		sig = y[trOut++];
		snprintf(dat, 30, "_%s_%04d.%03d.%02d.%02d.%02d", 
			ccname, hdr1->year, hdr1->yday, hdr1->hour, hdr1->min, hdr1->sec);
		snprintf(outfilename, 100, "%s.%s.%s.%s.%s.%s.%s.%s%s.sac", 
			hdr1->net, hdr1->sta, loc1, hdr1->chn, hdr2->net, hdr2->sta, loc2, hdr2->chn, dat);
		wrsac(outfilename, ccname, Lag1*dt, dt, sig, L, hdr1, hdr2);
	}
	
	return 0;
}

int StoreInBin (float **y, unsigned int L, unsigned int Tr, int Lag1, t_HeaderInfo *SacHeader1, 
		t_HeaderInfo *SacHeader2, float dt, char *filename, unsigned int *set, unsigned int *setin1, 
		unsigned int *setin2, unsigned int Trset, char *ccmethod, int verbose) {
	unsigned int tr;
	int nerr=0;
	t_ccheader *hdr=NULL;
	t_HeaderInfo *hdr1, *hdr2;
	time_t *time=NULL;
	float *data=NULL, *lag0=NULL;
	FILE *fid;
	
	if (verbose >= 2) {
		check_header (SacHeader1);
		check_header (SacHeader2);
	}
	
	hdr = (t_ccheader *)calloc(1, sizeof(t_ccheader));
	time = (time_t *)calloc(Trset, sizeof(time_t));
	lag0 = (float *)calloc(Trset, sizeof(float));
	data = (float *)calloc(Trset*L, sizeof(float));
	if (hdr == NULL || time == NULL || lag0 == NULL || data == NULL) {
		printf("StoreInBin: Out of memory (hdr).\n");
		nerr = -4;
	} else {
		/** Header    **/
		hdr1 = &SacHeader1[setin1[0]];
		hdr2 = &SacHeader2[setin2[0]];
		memcpy(hdr->method, ccmethod, 8);
		/* Info station 1. */
		memcpy(hdr->net1, hdr1->net, 8);
		memcpy(hdr->sta1, hdr1->sta, 8);
		memcpy(hdr->loc1, hdr1->loc, 8);
		memcpy(hdr->chn1, hdr1->chn, 8);
		hdr->stlat1 = hdr1->stla;
		hdr->stlon1 = hdr1->stlo;
		hdr->stel1  = hdr1->stel;
		/* Info station 2. */
		memcpy(hdr->net2, hdr2->net, 8);
		memcpy(hdr->sta2, hdr2->sta, 8);
		memcpy(hdr->loc2, hdr2->loc, 8);
		memcpy(hdr->chn2, hdr2->chn, 8);
		hdr->stlat2 = hdr2->stla;
		hdr->stlon2 = hdr2->stlo;
		hdr->stel2  = hdr2->stel;
		/* General info */
		hdr->nlags   = L;
		hdr->nseq    = Trset;
		hdr->tlength = dt*(L-1);
		hdr->lag1    = dt*Lag1;
		hdr->lag2    = dt*(Lag1 + L-1);
		
		/** Time info **/
		for (tr=0; tr<Trset; tr++) {
			hdr1 = &SacHeader1[setin1[tr]];
			time[tr] = hdr1->t;
		}
		
		/** Lag0 info **/
		for (tr=0; tr<Trset; tr++) {
			lag0[tr] = difftime(hdr1->t, hdr2->t) + (double)(hdr1->msec - hdr2->msec)/1000 + (hdr1->b - hdr2->b);
		}
		
		/** Data      **/
		for (tr=0; tr<Trset; tr++)
			memcpy(data + tr*L, y[set[tr]], L*sizeof(float));
		
		/** Save info. **/
		if (NULL == (fid = fopen(filename, "w"))) {
			printf("\a StoreInBin: cannot create %s file\n", filename);
			nerr = -2;
		} else {
			if ( verbose >= 2 && check_binheader(hdr) ) printf("StoreInBin: The header of %s is corrupted\n", filename);
			fwrite (hdr, sizeof(t_ccheader), 1, fid);
			fwrite (time, sizeof(time_t), Trset, fid);
			fwrite (lag0, sizeof(float), Trset, fid); /* Turn on when all the codes using sac2bin.h are updated */
			fwrite (data, sizeof(float), Trset*L, fid);
			fclose (fid);
		}
	}
	
	free(hdr); 
	free(lag0);
	free(time);
	free(data);
	
	return nerr;
}

int StoreInManyBins (float **y, unsigned int L, unsigned int Tr, int Lag1, t_HeaderInfo *SacHeader1, 
		t_HeaderInfo *SacHeader2, float dt, char *ccname, char *prefix, int verbose) {
	char filename[80];
	unsigned int tr, *set, Trset;
	int nerr=0;
	t_HeaderInfo *hdr1, *hdr2;
	
	if (verbose >= 2) {
		check_header (SacHeader1);
		check_header (SacHeader2);
	}
	
	set = (unsigned int *)calloc(Tr, sizeof(unsigned int));
	if (set == NULL) nerr = -1;
	else {
		char *trread, *chn1, *chn2;
		unsigned int trnext;
		
		trread = (char *)calloc(Tr+1, sizeof(char));
		if (trread == NULL) nerr = -1;
		else {
			trnext = 0;
			while (trnext < Tr) {
				/* Initializations */
				Trset = 0;
				memset(set, 0, Tr*sizeof(unsigned int));
				hdr1 = &SacHeader1[trnext];
				hdr2 = &SacHeader2[trnext];
				chn1 = hdr1->chn;
				chn2 = hdr2->chn;
				
				/* Create a set of traces having the same channels. */
				for (tr=0; tr<Tr; tr++)
					if ( !strcmp(SacHeader1[tr].chn, chn1) && !strcmp(SacHeader2[tr].chn, chn2) ) {
						trread[tr] = 1;
						set[Trset++] = tr;
					}
				
				/* Save traces */
				if (prefix != NULL)
					snprintf(filename, 80, "%s.%s.%s%c.%s.%s%c_%s.bin", prefix, hdr1->sta, hdr1->loc, chn1[2], 
						hdr2->sta, hdr2->loc, chn2[2], ccname);
				else
					snprintf(filename, 80, "%s.%s%c.%s.%s%c_%s.bin", hdr1->sta, hdr1->loc, chn1[2], 
						hdr2->sta, hdr2->loc, chn2[2], ccname);
				StoreInBin (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, filename, set, set, set, Trset, ccname, verbose);
				
				/* Find the next correlations to be saved. */
				while (trread[trnext] != 0) trnext++;
			}
		}
		free(trread);
	}
	
	free(set);
	return nerr;
}

int wrsac(char *filename, char *kinst, float beg, float dt, float *y, int nsamp, t_HeaderInfo *ptr1, t_HeaderInfo *ptr2) {
	float dummy[nsamp], lag0;
	int nerr, lcalda=1;
	
	lag0 = difftime(ptr1->t, ptr2->t) + (double)(ptr1->msec - ptr2->msec)/1000 + (ptr1->b - ptr2->b);
	// beg += lag0;
	
	newhdr();
	setnhv ("npts",   &nsamp,       &nerr, strlen("npts"));
	setfhv ("delta",  &dt,          &nerr, strlen("delta"));
	setkhv ("kinst",   kinst,       &nerr, strlen("kinst"), (strlen(kinst) < 8) ? strlen(kinst) : 8 );
	if (!ptr1->nostloc && !ptr2->nostloc) {
		setfhv ("stla",   &ptr2->stla,  &nerr, strlen("stla"));
		setfhv ("stlo",   &ptr2->stlo,  &nerr, strlen("stlo"));
		setfhv ("stel",   &ptr2->stel,  &nerr, strlen("stel"));
		setfhv ("stdp",   &ptr2->stdp,  &nerr, strlen("stdp"));
		setkhv ("knetwk",  ptr2->net,   &nerr, strlen("knetwk"), (strlen(ptr2->net) < 8) ? strlen(ptr2->net) : 8 );
		setkhv ("kstnm",   ptr2->sta,   &nerr, strlen("kstnm"),  (strlen(ptr2->sta) < 8) ? strlen(ptr2->sta) : 8 );
		setkhv ("khole",   ptr2->loc,   &nerr, strlen("khole"),  (strlen(ptr2->loc) < 8) ? strlen(ptr2->loc) : 8 );
		setkhv ("kcmpnm",  ptr2->chn,   &nerr, strlen("kcmpnm"), (strlen(ptr2->chn) < 8) ? strlen(ptr2->chn) : 8 );
		
		setfhv ("evla",   &ptr1->stla,  &nerr, strlen("evla"));
		setfhv ("evlo",   &ptr1->stlo,  &nerr, strlen("evlo"));
		setfhv ("evel",   &ptr1->stel,  &nerr, strlen("evel"));
		setfhv ("evdp",   &ptr1->stdp,  &nerr, strlen("evdp"));
		setkhv ("kuser0",  ptr1->net,   &nerr, strlen("kuser0"), (strlen(ptr1->net) < 8) ? strlen(ptr1->net) : 8 );
		setkhv ("kevnm",   ptr1->sta,   &nerr, strlen("kevnm"),  (strlen(ptr1->sta) < 8) ? strlen(ptr1->sta) : 8 );
		setkhv ("kuser1",  ptr1->loc,   &nerr, strlen("kuser1"), (strlen(ptr1->loc) < 8) ? strlen(ptr1->loc) : 8 );
		setkhv ("kuser2",  ptr1->chn,   &nerr, strlen("kuser2"), (strlen(ptr1->chn) < 8) ? strlen(ptr1->chn) : 8 );
	}
	setlhv ("lcalda", &lcalda,      &nerr, strlen("lcalda"));
	setnhv ("nzyear", &ptr1->year,  &nerr, strlen("nzyear"));
	setnhv ("nzjday", &ptr1->yday,  &nerr, strlen("nzjday"));
	setnhv ("nzhour", &ptr1->hour,  &nerr, strlen("nzhour"));
	setnhv ("nzmin",  &ptr1->min,   &nerr, strlen("nzmin"));
	setnhv ("nzsec",  &ptr1->sec,   &nerr, strlen("nzsec"));
	setnhv ("nzmsec", &ptr1->msec,  &nerr, strlen("nzmsec"));
	setfhv ("b",     &beg,   &nerr, strlen("b"));
	setfhv ("o",     &lag0,  &nerr, strlen("o"));
	setkhv ("ko", "0LagTime", &nerr, strlen("ko"), 8);
	setihv ("iztype", "io", &nerr, strlen("iztype"), strlen("io"));
	
	wsac0(filename, dummy, y, &nerr, strlen(filename));
	if (nerr) printf("wrsac: Error writing %s file\n", filename);
	
	// printf("%4d:%03d msec1 = %d msec2 = %d delay = %f s\n", ptr1->year, ptr1->yday, ptr1->msec, ptr2->msec, lag0);
	
	return nerr;
}

void clipping (float *x, unsigned int N) {
	float fa1, fa2;
	unsigned int n;
	
	fa1 = 4*qabsmedian(x, N) / 0.6745;
	for (n=0; n<N; n++) {
		fa2 = x[n];
		if (fabsf(fa2) > fa1) 
			x[n] = (fa2 > fa1) ? fa1 : -fa1;
	}
}

int RemoveOutlierTraces (float **xOut[], t_HeaderInfo *SacHeader[], unsigned int *Tr0, float *std, float nstd) {
	unsigned int tr, Tr=*Tr0;
	float **x = *xOut;
	float fa1;
	t_HeaderInfo *phd = *SacHeader;
	
	if (std == NULL || xOut == NULL || SacHeader == NULL || Tr0 == NULL) {
		printf("RemoveOutlierTraces: One required variable is NULL.\n");
		return 1;
	}
	
	unsigned int nskip;
	
	fa1 = qmedian(std, Tr);
	if (fa1==0) printf("RemoveOutlierTraces: 0 average std!\n");
	
	nskip = 0;
	for (tr=0; tr<Tr; tr++) {
		if (std[tr] > nstd * fa1) {
			nskip++;
			fftw_free(x[tr]); x[tr] = NULL;
			printf("RemoveOutlierTraces: Skipping %s.%s.%s.%s, the std is %f times the average std!\n", 
				phd[tr].net, phd[tr].sta, phd[tr].loc, phd[tr].chn, std[tr]/fa1);
		} else if (nskip && tr+1 < Tr) {
			memcpy (&phd[tr-nskip+1], &phd[tr+1], sizeof(t_HeaderInfo));
			x[tr-nskip+1] = x[tr+1];
		}
	}
	Tr -= nskip;
	
	return 0;
}

/* Note that considers ms but works at seconds precision. */
int swap_t_elem(const void *x1, const void *x2) {
	t_elem *t1 = (t_elem *)x1, *t2 = (t_elem *)x2;
	
	return (int)(difftime(t1->time, t2->time) + (double)(t1->msec - t2->msec)/1000);
}

int SortTraces (float **xOut[], t_HeaderInfo *SacHeader[], unsigned int *Tr0) {
	unsigned int Tr=*Tr0, st, ST, *ind;
	int nerr=0;
	float **x = *xOut, **x_copy;
	t_HeaderInfo *hdr = *SacHeader, *hdr_copy;
	
	t_elem *elem;
	
	if (xOut == NULL || SacHeader == NULL || Tr0 == NULL) {
		puts("SortTraces: One required variable is NULL.");
		return 1;
	}
	
	ST = Tr;
	if (NULL == (elem = (t_elem *)malloc(ST * sizeof(t_elem)) )) {
		puts("SortTraces: Out of memory.");
		return 2;
	}
	
	/* Find the permutation sorting SacHeader. */
	for (st=0; st<ST; st++) {
		elem[st].ind  = st;
		elem[st].time = hdr[st].t;
		elem[st].msec = hdr[st].msec;
	}
	qsort(elem, ST, sizeof(t_elem), swap_t_elem);
	
	/* Sort x & SacHeaer */
	if (NULL == (ind = (unsigned int *)malloc(ST*sizeof(unsigned int)) )) {
		puts("SortTraces: Out of memory.");
		free(elem);
		return 2;
	}
	
	for (st=0; st<ST; st++) ind[st] = elem[st].ind;
	free(elem);
	
	hdr_copy = (t_HeaderInfo *)malloc(Tr*sizeof(t_HeaderInfo));
	x_copy = (float **)malloc(Tr*sizeof(float *));
	if (hdr_copy == NULL && x_copy == NULL) { 
		puts("SortTraces: Out of memory.");
		nerr = 2;
	} else {
		memcpy(x_copy,   x,   Tr*sizeof(float *));
		memcpy(hdr_copy, hdr, Tr*sizeof(t_HeaderInfo));
		for (st=0; st<ST; st++) {
			x[st] = x_copy[ind[st]];
			memcpy(hdr + st, hdr_copy + ind[st], sizeof(t_HeaderInfo));
		}
	}
	
	free(x_copy);
	free(hdr_copy);
	free(ind);
	
	return nerr;
}

int MakePairedLists (float **xOut1[], t_HeaderInfo *SacHeader1[], unsigned int *TrOut1, 
                 float **xOut2[], t_HeaderInfo *SacHeader2[], unsigned int *TrOut2) {
	unsigned int tr1, tr2, Tr1=*TrOut1, Tr2=*TrOut2, TrOut, st1, st2, ST1, ST2, nskip1, nskip2;
	float **x1 = *xOut1, **x2 = *xOut2;
	t_HeaderInfo *hdr1 = *SacHeader1, *hdr2 = *SacHeader2;
	double td;
	
	if (xOut1 == NULL || SacHeader1 == NULL || TrOut1 == NULL || xOut2 == NULL || SacHeader2 == NULL || TrOut2 == NULL) {
		puts("MakePairedLists: One required variable is NULL.");
		return 1;
	}
	
	st1 = 0; st2 = 0; 
	ST1 = Tr1;
	ST2 = Tr2;
	nskip1 = 0; nskip2 = 0;
	while (st1 < ST1 && st2 < ST2) {
		td  = difftime(hdr1[st1].t, hdr2[st2].t);
		td += (double)(hdr1[st1].msec - hdr2[st2].msec)/1000;
		if (fabs(td) < 0.5) {   /* Keep the elements on both lists. */
			if (nskip1) {
				memcpy (&hdr1[st1-nskip1], &hdr1[st1], sizeof(t_HeaderInfo));
				x1[st1-nskip1] = x1[st1];
			}
			if (nskip2) {
				memcpy (&hdr2[st2-nskip2], &hdr2[st2], sizeof(t_HeaderInfo));
				x2[st2-nskip2] = x2[st2];
			}
			st1++; st2++;
		} else if (td < 0) { /* Remove element from list 1 if td < -0.5 */ 
			fftw_free(x1[st1]); 
			x1[st1] = NULL; 
			nskip1++; 
			st1++; 
		} else if (td > 0) { /* Remove element from list 2 if td >  0.5 */ 
			fftw_free(x2[st2]); 
			x2[st2] = NULL; 
			nskip2++; 
			st2++; 
		} 
	}
	
	TrOut = (Tr1-nskip1 < Tr2-nskip2) ? Tr1-nskip1 : Tr2-nskip2;
	for (tr1=TrOut; tr1<Tr1; tr1++) x1[tr1] = NULL;
	for (tr2=TrOut; tr2<Tr2; tr2++) x2[tr2] = NULL;
	
	*TrOut1 = TrOut;
	*TrOut2 = TrOut;
	
	return 0;
}

int CheckPairs (t_HeaderInfo *hdr1, unsigned int Tr1, t_HeaderInfo *hdr2, unsigned int Tr2) {
	unsigned int tr;
	double td;
	int nerr=0;
	
	if (hdr1 == NULL || hdr2 == NULL) {
		puts("CheckPairs: One required variable is NULL.");
		return -1;
	}
	
	if (Tr1 != Tr2) return -2;
	
	for (tr=0; tr<Tr1; tr++) {
		td  = difftime(hdr1[tr].t, hdr2[tr].t);
		td += (double)(hdr1[tr].msec - hdr2[tr].msec)/1000;
		if (fabs(td) > 0.5) nerr++;  /* KO. */
	}
	
	return nerr;
}

int AveWhite (float **x0, unsigned int N, unsigned int Tr, double freq[2], double dt) {
	fftw_plan pxX, pXx;
	double *x, *H, *HS, da1, mn, mx, win[11];
	float *pf1;
	fftw_complex *X;
	unsigned int Nz, Nh, tr, m, M, n, n1, n2;
	int nerr;
	
	Nz = 1 << (unsigned int)ceil(log2(N));
	Nh = Nz/2 + 1;  /* Number of complex used in r2c & c2r ffts. */
	x = (double *)fftw_malloc(Nz*sizeof(double));
	H = (double *)fftw_malloc(Nh*sizeof(double));
	HS = (double *)fftw_malloc(Nh*sizeof(double));
	X = (fftw_complex *)fftw_malloc(Nh*sizeof(fftw_complex));
	if (x != NULL && H != NULL && HS != NULL && X != NULL) {
		// fftw_init_threads();
		// fftw_plan_with_nthreads(omp_get_max_threads());
		
		pxX  = fftw_plan_dft_r2c_1d(Nz, x, X, FFTW_ESTIMATE); /* The FFT plan  */
		
		/** Average absolute spectrum **/
		for (tr=0; tr<Tr; tr++) {
			pf1 = x0[tr];
			for (n=0; n<N;  n++) x[n] = (double)pf1[n];
			for (n=N; n<Nz; n++) x[n] = 0;
			fftw_execute(pxX);
			
			for (n=0; n<Nh; n++) H[n] += cabs(X[n]);
		}
		da1 = 1./(double)Tr;
		for (n=0; n<Nh; n++) H[n] *= da1;
		
		/** Whitening filter **/
		n1 = freq[0]*dt*(double)Nz;
		n2 = freq[1]*dt*(double)Nz;
		da1 = 1./H[n1];
		for (n=0; n<n1; n++) H[n] = da1;
		for (   ; n<n2; n++) H[n] = 1./H[n];
		da1 = 1./H[n2];
		for (   ; n<Nh; n++) H[n] = da1;
		
		/* Max amplification of 100 times of the minimum value. */
		mn = mx = H[n1];
		for (n=n1+1; n<n2; n++) {
			da1 = H[n];
			if (da1 > mx) mx = da1;
			else if (da1 < mn) mn = da1;
		}
		if (mx > 100*mn) mx = 100*mn;
		for (n=0; n<Nh; n++)
			if (H[n] > mx) H[n] = mx;
		da1 = 1/(mn * (double)Nz); /* 1/mn so min gain is 1, and 1/Nz to normalize the ifft */
		for (n=0; n<Nh; n++) H[n] *= da1;
		
		/* Sprectrum smoothing using the blackman window. */
		for (m=0; m<11; m++) {
			da1 = 2*PI*m/(N-1);
			win[m] = 0.42 - 0.5*cos(da1) + 0.08*cos(2*da1);
		}
		da1 = 0;
		for (m=0; m<11; m++) da1 += win[m];
		da1 = 1/da1;
		for (m=0; m<11; m++) win[m] *= da1;
		
		/* Convolution with mirroring (DC and Nyquist samples are considered only once) */
		for (n=5; n<10; n++) {
			da1 = 0;
			for (m=0; m<=n; m++) da1 += win[m]*H[n-m];
			for (   ; m<11; m++) da1 += win[m]*H[m-n];
			HS[n-5] = da1;
		}
		for (n=11; n<Nh; n++) {
			da1 = 0;
			for (m=0; m<11; m++) da1 += win[m]*H[n-m];
			HS[n-5] = da1;
		}
		for (n=Nh; n<Nh+5; n++) {
			da1 = 0;
			M = n-(Nh-1);
			for (m=0; m<M;  m++) da1 += win[m]*H[Nh-1+m-M];
			for (   ; m<11; m++) da1 += win[m]*H[n-m];
			HS[n-5] = da1;
		}
		
		/** Do the whitening **/
		pXx = fftw_plan_dft_c2r_1d(Nz, X, x, FFTW_ESTIMATE); /* The IFFT plan */
		for (tr=0; tr<Tr; tr++) {
			pf1 = x0[tr];
			for (n=0; n<N;  n++) x[n] = (double)pf1[n];
			for (n=N; n<Nz; n++) x[n] = 0;
			fftw_execute(pxX);
			for (n=0; n<Nh; n++) X[n] *= HS[n];
			fftw_execute(pXx);
			for (n=0; n<N; n++) pf1[n] = (float)x[n];
		}
		
		fftw_destroy_plan(pxX);
		fftw_destroy_plan(pXx);
		
		// fftw_cleanup_threads();

	} 
	else nerr = -1;
	
	fftw_free(X);
	fftw_free(HS);
	fftw_free(H);
	fftw_free(x);
	
	return nerr;
}

float qabsmedian(float input[], int n) {
	register int i,j,l,m;
	register float x, t, out, *a;
	int k = n/2;
	
	if (NULL == (a = (float *)malloc(n * sizeof(float)) )) return 0;
	for (i=0; i<n; i++) a[i] = fabsf(input[i]);
	
	l = 0; m = n-1;
	while (l <m) {
		x = a[k];
		i = l; j = m;
		do {
			while (a[i] < x) i++;
			while (x < a[j]) j--;
			if ( i<=j) {
				t = a[i]; 
				a[i] = a[j]; 
				a[j] = t;
				i++; j--;
			}
		} while (i <= j);
		if (j < k) l = i;
		if (k < i) m = j;
	}

	if (n % 2) out = a[k];
	else {
		t = -FLT_MAX;
		for (i=0; i<k ; i++)
			if (a[i] <= a[k] && t < a[i]) t = a[i];
		out = (a[k] + t)/2;
	}
	free(a);
	return out;
}

float qmedian(float input[], int n) {
	register int i,j,l,m;
	register float x, t, out, *a;
	int k = n/2;
	
	if (NULL == (a = (float *)malloc(n * sizeof(float)) )) return 0;
	for (i=0; i<n; i++) a[i] = input[i];
	
	l = 0; m = n-1;
	while (l <m) {
		x = a[k];
		i = l; j = m;
		do {
			while (a[i] < x) i++;
			while (x < a[j]) j--;
			if ( i<=j) {
				t = a[i]; 
				a[i] = a[j]; 
				a[j] = t;
				i++; j--;
			}
		} while (i <= j);
		if (j < k) l = i;
		if (k < i) m = j;
	}

	if (n % 2) out = a[k];
	else {
		t = -FLT_MAX;
		for (i=0; i<k ; i++)
			if (a[i] <= a[k] && t < a[i]) t = a[i];
		out = (a[k] + t)/2;
	}
	free(a);
	return out;
}

int RDint (int * const x, const char *str) {
	char *pstr;
	
	*x = strtol(str, &pstr, 10);
	return (str == pstr) ? 1 : 0; 
}

int RDuint (unsigned int * const x, const char *str) {
	char *pstr;
	int ia1;
	
	ia1 = strtol(str, &pstr, 10);
	*x = (unsigned)abs(ia1);
	if (ia1 < 0 ) return -1;
	return (str == pstr) ? 1 : 0; 
}

int RDdouble (double * const x, const char *str) {
	char *pstr;
	
	*x = strtod(str, &pstr);
	return (str == pstr) ? 1 : 0; 
}

int RDdouble_array (double * const x, char * const str, unsigned int N) {
	unsigned int n;
	char *str0=str, *str1;
	
	for (n=0; n<N; n++) {
		x[n] = strtod(str0, &str1);
		if (str0 == str1) return 1;
		if (str0 != NULL) str0 = str1+1;
	}
	return 0;
}
