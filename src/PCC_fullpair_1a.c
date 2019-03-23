/*****************************************************************************/
/* Provided a list of stations, this program computes the cross-correlations */
/* between all possible pairs.                                               */
/*                                                                           */
/* Authors: Sergi Ventosa Rahuet (sergiventosa@hotmail.com)                  */
/*                                                                           */
/* Main features:                                                            */
/*   Computes 3 types of correlations:                                       */
/*    - The standard geometrically-normalized cross-correlations (GNCC),     */
/*    - the 1-bit amplitude normalization followed by the GNCC (1-bit GNCC). */
/*    - and the phase cross-correlation (PCC).                               */
/*                                                                           */
/*****************************************************************************/
/* Temporal: Save loc '' of Geoscope (G) stations as '00'                    */ 
/*****************************************************************************/

#include <complex.h>  /* When done before fftw3.h, makes fftw3.h use C99 complex types. */
#include <fftw3.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <sacio.h>
#include <float.h>
#include "FFTapps.h"
#include "ReadManySacs.h"
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
	unsigned int  ccgn;
	unsigned int  cc1b;
	unsigned int  clip;
	double        std;
	double        mindist;  /* Min distance in degrees. */ 
	double        maxdist;  /* Max distance in degrees.  */ 
	double        awhite[2];
	char          *fin1;
	char          *fin2;
} t_PCCmatrix;

typedef struct {
	unsigned int  ind;
	time_t        time;
} t_elem;

int PCCfullpair_main (t_PCCmatrix *fpcc);

int StoreInManySacs (double **y, unsigned int L, unsigned int Tr, int Lag1, t_HeaderInfo *SacHeader1, 
	t_HeaderInfo *SacHeader2, float dt, char *ccname);
int StoreInManyBins (double **y, unsigned int L, unsigned int Tr, int Lag1,  t_HeaderInfo *SacHeader1, 
	t_HeaderInfo *SacHeader2, float dt, char *ccname);
int wrsac(char *filename, char *kstnm, float beg, float dt, float *y, int nsamp, t_HeaderInfo *ptr1, t_HeaderInfo *ptr2);

void infooo();
void usage();
void clipping (double *x, unsigned int N);
int RemoveOutlierTraces (double **xOut[], t_HeaderInfo *SacHeader[], unsigned int *Tr0, float *std, float nstd);
int SortTraces (double **xOut[], t_HeaderInfo *SacHeader[], unsigned int *Tr0);
int MakePairedLists (double **xOut1[], t_HeaderInfo *SacHeader1[], unsigned int *TrOut1, 
	double **xOut2[], t_HeaderInfo *SacHeader2[], unsigned int *TrOut2);
int AveWhite (double **x0, unsigned int N, unsigned int Tr, double freq[2], double dt);
float qmedian(float a[], int n);
float qabsmedian(double input[], int n);

int RDint    (int * const x, const char *str);
int RDdouble (double * const x, const char *str);
int RDdouble_array (double * const x, char * const str, unsigned int N);

/* Main function: Mainly reading parameters. */
int main(int argc, char *argv[]) {
	t_PCCmatrix fpcc = {0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, {0, 0}, NULL, NULL};  /* Default parameters. */
	int i, er;

	if (argc < 3) {
		usage();
		return 0;
	}
	
	fpcc.fin1 = argv[1];
	fpcc.fin2 = argv[2];
	if (!strncmp(fpcc.fin1, "info", 4)) {
		infooo();
		return 0;
	}
	
	for (i=1; i<argc; i++) {
		if (!strncmp(argv[i], "tl1=", 4))      er = RDdouble(&fpcc.tl1, argv[i] + 4);
		else if (!strncmp(argv[i], "tl2=",   4)) er = RDdouble(&fpcc.tl2, argv[i] + 4);
		else if (!strncmp(argv[i], "v=",     2)) er = RDdouble(&fpcc.v,   argv[i] + 2);
		else if (!strncmp(argv[i], "nl1=",   4)) er = RDint(&fpcc.nl1,   argv[i] + 4);
		else if (!strncmp(argv[i], "nl2=",   4)) er = RDint(&fpcc.nl2,   argv[i] + 4);
		else if (!strncmp(argv[i], "isac",   4)) fpcc.iformat = 1;
		else if (!strncmp(argv[i], "imsacs", 6)) fpcc.iformat = 2;
		else if (!strncmp(argv[i], "osac",   4)) fpcc.oformat = 1;
		else if (!strncmp(argv[i], "obin",   4)) fpcc.oformat = 2;
		else if (!strncmp(argv[i], "pcc",    3)) fpcc.pcc  = 1;
		else if (!strncmp(argv[i], "ccgn",   4)) fpcc.ccgn = 1;
		else if (!strncmp(argv[i], "cc1b",   4)) fpcc.cc1b = 1;
		else if (!strncmp(argv[i], "clip",   4)) fpcc.clip = 1;
		else if (!strncmp(argv[i], "std=",   4)) er = RDdouble(&fpcc.std, argv[i] + 4);
		else if (!strncmp(argv[i], "mindist=", 8)) er = RDdouble(&fpcc.mindist, argv[i] + 8);
		else if (!strncmp(argv[i], "maxdist=", 8)) er = RDdouble(&fpcc.maxdist, argv[i] + 8);
		else if (!strncmp(argv[i], "awhite=",7)) er = RDdouble_array(fpcc.awhite, argv[i] + 7, 2);
		else if (!strncmp(argv[i], "info",   4)) {
			infooo();
			return 0;
		}
	}
	er = PCCfullpair_main(&fpcc); /* The one who make the job. */
	return er;
}

int PCCfullpair_main (t_PCCmatrix *fpcc) {
	t_HeaderInfo *SacHeader1=NULL, *SacHeader2=NULL;
	float *std=NULL;
	float dt, dt1;
	double **x1=NULL, **x2=NULL, **y=NULL, *px, da2;
	double lat1=-90, lon1=0, lat2=90, lon2=0, gcarc;
	unsigned int tr, Tr, Tr1, Tr2, n, N, N1;
	int Lag1, Lag2, ia1, L, nerr=0;
	char nickpcc[16]; /* Up to the first 8 are saved in the sac header. */
	
	/* Input checkings */
	if (fpcc == NULL) {       printf("PCCfullpair_main: NULL input\n");           return -1; }
	if (fpcc->fin1 == NULL) { printf("PCCfullpair_main: NULL input filename1\n"); return -1; }
	if (fpcc->fin2 == NULL) { printf("PCCfullpair_main: NULL input filename2\n"); return -1; }
	
	/* Read input files */
	if (fpcc->iformat == 1) {
		if ( 0 != ReadLocation (&lat1, &lon1, fpcc->fin1) ) return 0;
		if ( 0 != ReadLocation (&lat2, &lon2, fpcc->fin2) ) return 0;
	} else if (fpcc->iformat == 2) { 
		if ( 0 != ReadLocation_ManySacsFile (&lat1, &lon1, fpcc->fin1) ) return 0;
		if ( 0 != ReadLocation_ManySacsFile (&lat2, &lon2, fpcc->fin2) ) return 0;
	} else { printf("PCCfullpair_main: Unknown format."); nerr = 5; }
	
	lat1 *= DEG2RAD;
	lon1 *= DEG2RAD;
	lat2 *= DEG2RAD;
	lon2 *= DEG2RAD;
	sph_gcarc (&gcarc, lat1, lon1, &lat2, &lon2, 1);
	gcarc *= RAD2DEG;
	
	if (fpcc->mindist > 0 || fpcc->maxdist > 0) { /* Check minimum distance */
		if (fpcc->mindist > 0 && gcarc < fpcc->mindist) {
			printf("The interstation distance is smaller than the %f degrees requiered on %s - %s.\n", fpcc->mindist, fpcc->fin1, fpcc->fin2);
			return 0;
		}
		
		if (fpcc->maxdist > 0 && gcarc > fpcc->maxdist) {
			printf("The interstation distance is higher than the %f degrees requiered on %s - %s.\n", fpcc->maxdist, fpcc->fin1, fpcc->fin2);
			return 0;
		}
	}
	
	if (fpcc->iformat == 1)      nerr = ReadManySacs (&x1, &SacHeader1, &Tr1, &N1, &dt1, fpcc->fin1);
	else if (fpcc->iformat == 2) nerr = Read_ManySacsFile (&x1, &SacHeader1, &Tr1, &N1, &dt1, fpcc->fin1);
	else { printf("PCCfullpair_main: Unknown format."); nerr = 5; }
	if (nerr) printf("PCCfullpair_main: Something when wrong in ReadManySacs when reading the first filelist! (nerr = %d)\n", nerr);
	if (Tr1==0) { printf("PCCfullpair_main: %s is empty.", fpcc->fin1); return 0; }
	
	if (fpcc->iformat == 1)      nerr = ReadManySacs (&x2, &SacHeader2, &Tr2, &N, &dt, fpcc->fin2);
	else if (fpcc->iformat == 2) nerr = Read_ManySacsFile (&x2, &SacHeader2, &Tr2, &N, &dt, fpcc->fin2);
	else { printf("PCCfullpair_main: Unknown format."); nerr = 5; }
	if (nerr) printf("PCCfullpair_main: Something when wrong in ReadManySacs when reading the second filelist! (nerr = %d)\n", nerr);
	if (Tr2==0) { printf("PCCfullpair_main: %s is empty.", fpcc->fin2); return 0; }
	if (N1 != N) printf("PCCfullpair_main: Traces of these stations has different lengths: %u:%u\n", N1, N);
	if (dt1 != dt) printf("PCCfullpair_main: Traces of these stations has different samplings: %f:%f\n", dt1, dt);
	
	nerr = RemoveZeroTraces (&x1, &SacHeader1, &Tr1, N);
	if (nerr) printf("PCCfullpair_main: Something when wrong in RemoveZeroTraces of station 1! (nerr = %d)\n", nerr);
	nerr = RemoveZeroTraces (&x2, &SacHeader2, &Tr2, N);
	if (nerr) printf("PCCfullpair_main: Something when wrong in RemoveZeroTraces of station 2! (nerr = %d)\n", nerr);
	
	/* Outliers and clippling for station 1 */
	/* Data std. (The signal should have no mean). */
	if (fpcc->std) {
		if (NULL == (std = (float *)calloc(Tr1, sizeof(float)) )) {
			Destroy_DoubleArrayList(x1, Tr1);
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
		if (nerr) printf("PCCfullpair_main: Something when wrong in RemoveOutlierTraces of station 1! (nerr = %d)\n", nerr);
	}
	
	/* Outliers and clippling for station 2 */
	if (fpcc->std) {
		if (NULL == (std = (float *)calloc(Tr2, sizeof(float)) )) {
			Destroy_DoubleArrayList(x2, Tr2);
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
		if (nerr) printf("PCCfullpair_main: Something when wrong in RemoveOutlierTraces of station 2! (nerr = %d)\n", nerr);
		free(std);
	}
	
	/* Sort out each list. */
	SortTraces (&x1, &SacHeader1, &Tr1);
	SortTraces (&x2, &SacHeader2, &Tr2);
	
	/* Remove time sequences not present on both stations. */
	MakePairedLists (&x1, &SacHeader1, &Tr1, &x2, &SacHeader2, &Tr2);
	Tr = Tr1;
	
	if (fpcc->awhite[0] > 0 && fpcc->awhite[1] > fpcc->awhite[0]) {
		AveWhite (x1, N, Tr, fpcc->awhite, dt);
		AveWhite (x2, N, Tr, fpcc->awhite, dt);
	}
	
	/* Calculate lags. */
	if (fpcc->nl1) Lag1 = fpcc->nl1;
	else if (fpcc->tl1) Lag1 = fpcc->tl1 / dt;
	else Lag1 = 0;
	if (fpcc->nl2) Lag2 = fpcc->nl2;
	else if (fpcc->tl1) Lag2 = fpcc->tl2 / dt;
	else Lag2 = 0;
	
	if (Lag1 > Lag2) { ia1 = Lag1; Lag1 = Lag2; Lag2 = ia1; }
	if (abs(Lag1) > N || abs(Lag2) > N) printf("PCCfullpair_main: Lags exceed the sequence length\n");
	L = Lag2-Lag1+1;
	
	printf("Lag1 = %d, Lag2 = %d, L = %d, N = %d, Tr = %d, gcarc = %f\n", Lag1, Lag2, L, N, Tr, gcarc);
	
	if (fpcc->v==2)      strcpy(nickpcc, "pcc2");
	else if (fpcc->v==1) strcpy(nickpcc, "pcc1");
	else sprintf(nickpcc, "pcc%.1f", fpcc->v);
	
	/* Output array memory */
	if (NULL == (y = Create_DoubleArrayList (L, Tr) )) {
		printf ("PCCfullpair_main: Out of memory on the output array (%d x %d)\n", L, Tr);
		nerr = 4;
	} else {
		/* The actual cross-correlations */
		if (fpcc->pcc) {  /* PCCs: */
			if (fpcc->v==2)      pcc2_set (y, x1, x2, N, Tr, Lag1, Lag2);
			else if (fpcc->v==1) pcc1_set (y, x1, x2, N, Tr, Lag1, Lag2);
			else pcc_set (y, x1, x2, N, Tr, fpcc->v, Lag1, Lag2);
			if (fpcc->oformat==1) 
				StoreInManySacs (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, nickpcc);
			else if (fpcc->oformat==2) 
				StoreInManyBins (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, nickpcc);
		}
		
		if (fpcc->ccgn) {  /* GNCCs */
			ccgn_set (y, x1, x2, N, Tr, Lag1, Lag2);
			if (fpcc->oformat==1) 
				StoreInManySacs (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, "ccgn");
			else if (fpcc->oformat==2) 
				StoreInManyBins (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, "ccgn");
		}
		
		if (fpcc->cc1b) {  /* 1-bit + GNCCs */
			cc1b_set (y, x1, x2, N, Tr, Lag1, Lag2);
			if (fpcc->oformat==1) 
				StoreInManySacs (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, "cc1b");
			else if (fpcc->oformat==2) 
				StoreInManyBins (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, "cc1b");
		}
		
		Destroy_DoubleArrayList (y, Tr);
	}
	
	Destroy_DoubleArrayList (x1, Tr);
	Destroy_DoubleArrayList (x2, Tr);
	free (SacHeader1);
	free (SacHeader2);
	
	return 0;
}

void infooo() {
	puts("\nProvided a list of stations, this program computes the geometrically-normalized and phase cross-correlations (CCGN & PCC) between all possible pairs.");
	puts("I developed this program for Ventosa et al. (2017). Information on the PCC is published in Schimmel (1999).\n");
	puts("Schimmel M. 1999.  Phase cross-correlations: Design, comparisons, and applications, Bulletin of the Seismological Society of America, 89(5), 1366-1378.");
	puts("Ventosa, S., Schimmel, M., & Stutzmann, E., 2017. Extracting surface waves, hum and normal modes: Time-scale phase-weighted stack and beyond, Geophysical Journal International, 211, 30-44, doi:10.1093/gji/ggx284");
	puts("Ventosa, S., Schimmel, M., & Stutzmann, E., 2019. Towards the processing of large data volumes with phase cross-correlation, Seismological Research Letters."); 
	puts("AUTHOR: Sergi Ventosa Rahuet (sergiventosa(at)hotmail.com)");
	puts("Last modification: 21/03/2019\n");
}

void usage() {
	puts("\nProvided the available traces from two stations, this program computes cross-correlations between them.");
	puts("");
	puts("USAGE: PCC_fullpair_1a filelist1 filelist2 parameters");
	puts("  filelist1:  text file containing a list of (SAC) files of station 1. One file name per");
	puts("              line, traces must have same begin (b), number of samples (nsmpl) and sampling");
	puts("              interval (dt). The date and time header information is needed to properly");
	puts("              pair the traces of the two stations.");
	puts("  filelist2:  idem to filelist1 but for station 2.");
	puts("  parameters: provided in arbitrary order without any blanck around '=' .");
	puts("");
	puts("  Data can also be saved in a single file per station in the msacs format (see the example below).");
	puts("");
	puts("Most commonly used parameters");
	puts("  nl1=   : stating sample lag. nl1=0 by default.");
	puts("  nl2=   : ending sample lag. nl2=0 by default.");
	puts("  tl1=   : stating relative time lag in seconds. Modifies nl1. tl1=0 by default.");
	puts("  tl2=   : ending relative time lag in seconds. Modifies nl2. tl2=0 by default.");
	puts("  ccgn   : compute geometrically normalized cross-correlation. Not computed by default.");
	puts("  cc1b   : compute 1-bit amplitude normalization followed by ccgn. Not computed by default.");
	puts("  pcc    : compute phase cross-correlation. Not computed by default.");
	puts("  v      : pcc power, sum(|a+b|^v - |a+b|^v). Default is v=2");
	puts("  info   : write background and main references to screen.");
	puts("           Just type: PCC_fullpair info.");
	puts("");
	puts("Input/Output data format");
	puts("  isac   : Input traces are in the SAC format (default)");
	puts("  imsacs : Input traces are saved in two files, one per station.");
	puts("  osac   : Output interstation correlations are saved in many files in the SAC format (default)");
	puts("  obin   : Output interstation correlations are saved in one file that can be read by the time-");
	puts("           scale phase-weighted stack code ( https://github.com/sergiventosa/ts-PWS ).");
	puts("");
	puts("EXAMPLES");
	puts("  PCC of power 1 and CCGN between all pairs among the traces listed in filelist1.txt and");
	puts("  filelist2.txt:");
	puts("     PCC_fullpair_1a filelist1.txt filelist2.txt tl1=-1000 tl2=1000 ccgn pcc v=1");
	puts("");
	puts("  PCC of power 2 and 1-bit CCGN between all pairs among the traces stored in sta1.msacs");
	puts("  and sta2.msacs:");
	puts("     Filelist2msacs filelist1.txt sta1.msacs");
	puts("     Filelist2msacs filelist2.txt sta2.msacs");
	puts("     PCC_fullpair sta1.msacs sta2.msacs imsacs tl1=-1000 tl2=1000 cc1b pcc v=2");
	puts("");
	puts("AUTHOR: Sergi Ventosa, 21/03/2019");
	puts("Please, do not hesitate to send bugs, comments or improvements to sergiventosa(at)hotmail.com\n");
}

int StoreInManySacs (double **y, unsigned int L, unsigned int Tr, int Lag1, t_HeaderInfo *SacHeader1, 
		t_HeaderInfo *SacHeader2, float dt, char *ccname) {
	char dat[30], outfilename[80], loc1[9], loc2[9];
	t_HeaderInfo *hdr1, *hdr2;
	unsigned int n, tr1, trOut;
	double *pd1;
	float *sig;
	
	if (NULL == (sig = (float *)calloc(L, sizeof(float)) )) return 1;
	
	trOut = 0;
	for (tr1=0; tr1<Tr; tr1++) {
		hdr1 = &SacHeader1[tr1];
		hdr2 = &SacHeader2[tr1];
		strncpy(loc1, hdr1->loc, 9);
		if ( !strncmp(hdr1->net, "G", 4) && !strcmp(loc1, "") ) strcpy(loc1, "00");
		strncpy(loc2, hdr2->loc, 9);
		if ( !strncmp(hdr2->net, "G", 4) && !strcmp(loc2, "") ) strcpy(loc2, "00");
		
		pd1 = y[trOut++];
		for (n=0; n<L; n++) sig[n] = (float)pd1[n];
		snprintf(dat, 30, "_%s_%04d.%03d.%02d.%02d.%02d", 
			ccname, hdr1->year, hdr1->yday, hdr1->hour, hdr1->min, hdr1->sec);
		snprintf(outfilename, 80, "%s.%s%c.%s.%s%c%s.sac", 
			hdr1->sta, loc1, hdr1->chn[2], hdr2->sta, loc2, hdr2->chn[2], dat);
		wrsac(outfilename, ccname, Lag1*dt, dt, sig, L, hdr1, hdr2);
	}
	
	free(sig);
	return 0;
}

int StoreInBin (double **y, unsigned int L, unsigned int Tr, int Lag1, t_HeaderInfo *SacHeader1, 
		t_HeaderInfo *SacHeader2, float dt, char *filename, unsigned int *set, unsigned int *setin1, unsigned int *setin2, unsigned int Trset, char *ccmethod) {
	unsigned int tr, l;
	int nerr=0;
	t_ccheader *hdr=NULL;
	t_HeaderInfo *hdr1, *hdr2;
	time_t *time=NULL;
	float *data=NULL, *pf1;
	double *pd1;
	FILE *fid;
	
	hdr = (t_ccheader *)calloc(1, sizeof(t_ccheader));
	time = (time_t *)calloc(Trset, sizeof(time_t));
	data = (float *)calloc(Trset*L, sizeof(float));
	if (hdr == NULL || time == NULL || data == NULL) {
		printf("StoreInBin: Out of memory (hdr).\n");
		nerr = -4;
	} else {
		/** Header    **/
		hdr1 = &SacHeader1[set[0]];
		hdr2 = &SacHeader2[set[0]];
		strncpy(hdr->method, ccmethod, 8);
		/* Info station 1. */
		strncpy(hdr->net1, hdr1->net, 8);
		strncpy(hdr->sta1, hdr1->sta, 8);
		strncpy(hdr->loc1, hdr1->loc, 8);
		strncpy(hdr->chn1, hdr1->chn, 8);
		hdr->stlat1 = hdr1->stla;
		hdr->stlon1 = hdr1->stlo;
		hdr->stel1  = hdr1->stel;
		/* Info station 2. */
		strncpy(hdr->net2, hdr2->net, 8);
		strncpy(hdr->sta2, hdr2->sta, 8);
		strncpy(hdr->loc2, hdr2->loc, 8);
		strncpy(hdr->chn2, hdr2->chn, 8);
		hdr->stlat2 = hdr2->stla;
		hdr->stlon2 = hdr2->stlo;
		hdr->stel2  = hdr2->stel;
		/* General info */
		hdr->nlags   = L;
		hdr->nseq    = Trset;
		hdr->tlength = L*dt;
		hdr->lag1    = dt*Lag1;
		hdr->lag2    = dt*(Lag1 + L-1);
		
		/** Time info **/
		for (tr=0; tr<Trset; tr++) {
			hdr1 = &SacHeader1[setin1[tr]];
			time[tr] = hdr1->t;
		}
		
		/** Data      **/
		for (tr=0; tr<Trset; tr++) {
			pd1 = y[set[tr]];
			pf1 = data + tr*L;
			for (l=0; l<L; l++) pf1[l] = (float)pd1[l];
		}
		
		/** Save info. **/
		if (NULL == (fid = fopen(filename, "w"))) {
			printf("\a StoreInBin: cannot create %s file\n", filename);
			nerr = -2;
		} else {
			fwrite (hdr, sizeof(t_ccheader), 1, fid);
			fwrite (time, sizeof(time_t), Trset, fid);
			fwrite (data, sizeof(float), Trset*L, fid);
			fclose (fid);
		}
	}
	
	free(hdr); 
	free(time); 
	free(data);
	
	return nerr;
}

int StoreInManyBins (double **y, unsigned int L, unsigned int Tr, int Lag1, t_HeaderInfo *SacHeader1, 
		t_HeaderInfo *SacHeader2, float dt, char *ccname) {
	char filename[80];
	unsigned int tr, *set, Trset;
	int nerr=0;
	t_HeaderInfo *hdr1, *hdr2;
	
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
				snprintf(filename, 80, "%s.%s%c.%s.%s%c_%s.bin", hdr1->sta, hdr1->loc, chn1[2], 
					hdr2->sta, hdr2->loc, chn2[2], ccname);
				StoreInBin (y, L, Tr, Lag1, SacHeader1, SacHeader2, dt, filename, set, set, set, Trset, ccname);
				
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
	float dummy[nsamp];
	int nerr, lcalda=1;
	
	newhdr();
	setnhv ("npts",   &nsamp,       &nerr, strlen("npts"));
	setfhv ("delta",  &dt,          &nerr, strlen("delta"));
	setkhv ("kinst",   kinst,       &nerr, strlen("kinst"), (strlen(kinst) < 8) ? strlen(kinst) : 8 );
	if (!ptr1->nostloc && !ptr2->nostloc) {
		setfhv ("stla",   &ptr2->stla,  &nerr, strlen("stla"));
		setfhv ("stlo",   &ptr2->stlo,  &nerr, strlen("stlo"));
		setfhv ("stel",   &ptr2->stel,  &nerr, strlen("stel"));
		setfhv ("stdp",   &ptr2->stdp,  &nerr, strlen("stdp"));
		setkhv ("knetwk",  ptr2->net,   &nerr, strlen("knetwk"), strlen(ptr2->net));
		setkhv ("kstnm",   ptr2->sta,   &nerr, strlen("kstnm"),  strlen(ptr2->sta));
		setkhv ("khole",   ptr2->loc,   &nerr, strlen("khole"),  strlen(ptr2->loc));
		setkhv ("kcmpnm",  ptr2->chn,   &nerr, strlen("kcmpnm"), strlen(ptr2->chn));
		
		setfhv ("evla",   &ptr1->stla,  &nerr, strlen("evla"));
		setfhv ("evlo",   &ptr1->stlo,  &nerr, strlen("evlo"));
		setkhv ("kuser0",  ptr1->net,   &nerr, strlen("kuser0"), strlen(ptr1->net));
		setkhv ("kevnm",   ptr1->sta,   &nerr, strlen("kevnm"),  strlen(ptr1->sta));
		setkhv ("kuser1",  ptr1->loc,   &nerr, strlen("kuser1"), strlen(ptr1->loc));
		setkhv ("kuser2",  ptr1->chn,   &nerr, strlen("kuser2"), strlen(ptr1->chn));
	}
	setlhv ("lcalda", &lcalda,      &nerr, strlen("lcalda"));
	setnhv ("nzyear", &ptr1->year,  &nerr, strlen("nzyear"));
	setnhv ("nzjday", &ptr1->yday,  &nerr, strlen("nzjday"));
	setnhv ("nzhour", &ptr1->hour,  &nerr, strlen("nzhour"));
	setnhv ("nzmin",  &ptr1->min,   &nerr, strlen("nzmin"));
	setnhv ("nzsec",  &ptr1->sec,   &nerr, strlen("nzsec"));
	setnhv ("nzmsec", &ptr1->msec,  &nerr, strlen("nzmsec"));
	setfhv ("b",     &beg,   &nerr, strlen("b"));
	setihv ("iztype", "io",  &nerr, strlen("iztype"), strlen("io"));
	
	wsac0(filename, dummy, y, &nerr, strlen(filename));
	if (nerr) printf("wrsac: Error writing %s file\n", filename);
	return nerr;
}

void clipping (double *x, unsigned int N) {
	float fa1, fa2;
	unsigned int n;
	
	fa1 = 4*qabsmedian(x, N) / 0.6745;
	for (n=0; n<N; n++) {
		fa2 = (float)x[n];
		if (fabsf(fa2) > fa1) 
			x[n] = (double)((fa2 > fa1) ? fa1 : -fa1);
	}
}

int RemoveOutlierTraces (double **xOut[], t_HeaderInfo *SacHeader[], unsigned int *Tr0, float *std, float nstd) {
	unsigned int tr, Tr=*Tr0;
	double **x = *xOut;
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
			printf("RemoveOutlierTraces: Skipping %s.%s.%s.%s, the std is %f times the average!\n", 
				phd[tr].net, phd[tr].sta, phd[tr].loc, phd[tr].chn, std[tr]/fa1);
		} else if (nskip && tr+1 < Tr) {
			memcpy (&phd[tr-nskip+1], &phd[tr+1], sizeof(t_HeaderInfo));
			x[tr-nskip+1] = x[tr+1];
		}
	}
	Tr -= nskip;
	
	return 0;
}

/* Note that ms are ignored. */
int swap_t_elem(const void *x1, const void *x2) {
	t_elem *t1 = (t_elem *)x1, *t2 = (t_elem *)x2;
	
	return (int)difftime(t1->time, t2->time);
}

int SortTraces (double **xOut[], t_HeaderInfo *SacHeader[], unsigned int *Tr0) {
	unsigned int Tr=*Tr0, st, ST, *ind;
	int nerr=0;
	double **x = *xOut, **x_copy;
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
	x_copy = (double **)malloc(Tr*sizeof(double *));
	if (hdr_copy == NULL && x_copy == NULL) { 
		puts("SortTraces: Out of memory.");
		nerr = 2;
	} else {
		memcpy(x_copy,   x,   Tr*sizeof(double *));
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

int MakePairedLists (double **xOut1[], t_HeaderInfo *SacHeader1[], unsigned int *TrOut1, 
                 double **xOut2[], t_HeaderInfo *SacHeader2[], unsigned int *TrOut2) {
	unsigned int tr1, tr2, Tr1=*TrOut1, Tr2=*TrOut2, TrOut, st1, st2, ST1, ST2, nskip1, nskip2;
	double **x1 = *xOut1, **x2 = *xOut2;
	t_HeaderInfo *hdr1 = *SacHeader1, *hdr2 = *SacHeader2;
	time_t td;
	
	if (xOut1 == NULL || SacHeader1 == NULL || TrOut1 == NULL || xOut2 == NULL || SacHeader2 == NULL || TrOut2 == NULL) {
		puts("MakePairedLists: One required variable is NULL.");
		return 1;
	}
	
	st1 = 0; st2 = 0; 
	ST1 = Tr1;
	ST2 = Tr2;
	nskip1 = 0; nskip2 = 0;
	while (st1 < ST1 && st2 < ST2) {
		td = difftime(hdr1[st1].t, hdr2[st2].t);
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
		} else if (td < 0) { /* Remove element from list 1. */ 
			fftw_free(x1[st1]); 
			x1[st1] = NULL; 
			nskip1++; 
			st1++; 
		} else if (td > 0) { /* Remove element from list 2. */ 
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

int AveWhite (double **x0, unsigned int N, unsigned int Tr, double freq[2], double dt) {
	fftw_plan pxX, pXx;
	double *x, *H, *HS, da1, mn, mx, win[11];
	fftw_complex *X;
	unsigned int Nz, Nh, tr, m, M, n, n1, n2;
	int nerr;
	
	Nz = 1 << (unsigned int)ceil(log2(N));
	Nh = Nz/2 + 1;  /* Number of complex used in r2c & c2r ffts. */
	x = (double *)fftw_malloc(Nz*sizeof(double));
	H = (double *)fftw_malloc(Nh*sizeof(double));
	HS = (double *)fftw_malloc(Nh*sizeof(double));
	X = (fftw_complex *)fftw_malloc(Nh*sizeof(fftw_complex));
	if (x == NULL || H == NULL || HS == NULL || X == NULL)
		nerr = -1;
	else {
		pxX  = fftw_plan_dft_r2c_1d(Nz, x, X, FFTW_ESTIMATE); /* FFT plan  */
		
		/** Average absolute spectrum **/
		for (tr=0; tr<Tr; tr++) {
			memcpy(x, x0[tr], N*sizeof(double));
			for (n=N; n<Nz; n++) x[n] = 0;
			fftw_execute(pxX);
			
			for (n=0; n<Nh; n++) H[n] += cabs(X[n]);
		}
		for (n=0; n<Nh; n++) H[n] /= Nz;
		
		/** Whitening filter **/ 
		n1 = freq[0]*dt*Nz/N;
		n2 = freq[1]*dt*Nz/N;
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
		
		/* Sprectrum smoothing using the blackman window. */
		for (m=0; m<11; m++) {
			da1 = 2*PI*m/(N-1);
			win[m] = 0.42 - 0.5*cos(da1) + 0.08*cos(2*da1);
		}
		
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
		pXx = fftw_plan_dft_c2r_1d(Nz, X, x, FFTW_ESTIMATE); /* IFFT plan */
		for (tr=0; tr<Tr; tr++) {
			memcpy(x, x0[tr], N*sizeof(double));
			for (n=N; n<Nz; n++) x[n] = 0;
			fftw_execute(pxX);
			for (n=0; n<Nh; n++) X[n] *= HS[n];
			fftw_execute(pXx);
			memcpy(x0[tr], x, N*sizeof(double));
		}
	}
	
	fftw_free(X);
	fftw_free(HS);
	fftw_free(H);
	fftw_free(x);
	return nerr;
}

float qabsmedian(double input[], int n) {
	register int i,j,l,m;
	register float x, t, out, *a;
	int k = n/2;
	
	if (NULL == (a = (float *)malloc(n * sizeof(float)) )) return 0;
	for (i=0; i<n; i++) a[i] = (float)fabs(input[i]);
	
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
