#ifndef FFTAPPS_H
#define FFTAPPS_H

#include <complex.h>

int AnalyticSignal (double complex *y, double *x, unsigned int N);
int xcorr (double complex *y, double complex *x1, double complex *x2, unsigned int N);
int xcorr_real (double *y, double *x1, double *x2, unsigned int N);
int PhaseSignal (double complex *y, double *x, unsigned int N);
int pcc_set  (double ** const y, double ** const x1, double ** const x2, const int N, const unsigned int Tr, const double v, const int Lag1, const int Lag2);
int pcc1_set (double ** const y, double ** const x1, double ** const x2, const int N, const unsigned int Tr, const int Lag1, const int Lag2);
int pcc2_set (double ** const y, double ** const x1, double ** const x2, const unsigned int N, const unsigned int Tr, const int Lag1, const int Lag2);
int ccgn_set (double ** const y, double ** const x1, double ** const x2, const unsigned int N, const unsigned int Tr, const int Lag1, const int Lag2);
int cc1b_set (double ** const y, double ** const x1, double ** const x2, const unsigned int N, const unsigned int Tr, const int Lag1, const int Lag2);

int gncc2 (double * const y, double * const x1, double * const x2, const unsigned int N, const int Lag1, const int Lag2);

#endif
