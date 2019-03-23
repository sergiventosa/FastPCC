#ifndef MANYSACS_H
#define MANYSACS_H

#include <stdint.h>
#include <time.h>

typedef struct {
	int32_t  npts;    /* Number of samples.    */
	float    dt;      /* Sampling period.      */
	float    b;
	int32_t  year;
	int32_t  yday;
	int32_t  hour;
	int32_t  min;
	int32_t  sec;
	int32_t  msec;
	time_t   t;
	float    stla;    /* Station latitud.      */
	float    stlo;    /* Station longitud.     */
	float    stel;    /* Station elevation     */
	float    stdp;    /* Station depth         */
	float    cmpaz;   /* Azimuth  (degrees clockwise from north) */
	float    cmpinc;  /* Inclination (degrees from vertical).    */
	char     sta[9];  /* Name of the station.  */
	char     net[9];  /* Network.              */
	char     chn[9];  /* Channel.              */
	char     loc[9];  /* Location/hole         */
	int32_t  nostloc;
	int32_t  nocmp;
} t_HeaderInfo;

typedef struct {
	char     FormatID[8];  /* FormatID: MSACS1                          */
	int32_t  nseq;          /* Number of sequences in the file.         */
	int32_t  npts;          /* Length of the sequences (0 if different) */
	int32_t  SingleChannel; /* 0 (different) / 1 (channels) channels    */ 
} t_HeaderManySacsBinary;

double **Destroy_DoubleArrayList (double **x, unsigned int Tr);
double **Create_DoubleArrayList (unsigned int N, unsigned int Tr);
int ReadLocation (double *stlat, double *stlon, char *fin);
int ReadManySacs (double **xOut[], t_HeaderInfo *SacHeaderOut[], unsigned int *TrOut, unsigned int *NOut, float *dtOut, char *fin);
int ReadManySacs_WithDiffLength (double **xOut[], t_HeaderInfo *SacHeaderOut[], unsigned int *TrOut, char *fin);
void DestroyFilelist (char *p[]);
int CreateFilelist (char **filename[], unsigned int *Tr, char *filelist);

int RemoveZeroTraces (double **x[], t_HeaderInfo *SacHeader[], unsigned int *Tr, unsigned int N);

int Read_ManySacsFile (double **xOut[], t_HeaderInfo *SacHeaderOut[], unsigned int *TrOut, unsigned int *NOut, float *dtOut, char *fin);
int Write_ManySacsFile (double *x[], t_HeaderInfo *SacHeader, unsigned int Tr, unsigned int N, char *fout);
int ReadLocation_ManySacsFile (double *stlat, double *stlon, char *fin);

#endif
