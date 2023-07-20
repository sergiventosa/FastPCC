#ifndef CCS_CUDA_H
#define CCS_CUDA_H

#include <complex.h>
#include <semaphore.h>

#ifdef __cplusplus
extern "C" int pcc_highlevel (float **y, _Complex float **x1, _Complex float **x2, int N, unsigned int Tr, float v, int Lag1, int Lag2, sem_t *anok);
extern "C" int pcc1_highlevel (float **y, _Complex float **x1, _Complex float **x2, int N, unsigned int Tr, int Lag1, int Lag2, sem_t *anok);
extern "C" void pcc1_highlevel2 (float **y, float **x1, float **x2, int N, unsigned int Tr, int Lag1, int Lag2);
extern "C" int cuda_tspcc2_set_freq (float **y0, float **x01, float **x02, unsigned int N, unsigned int Tr, unsigned int Nz, const int Lag1, const int Lag2, 
	unsigned int S, double *scale, _Complex double **wc, unsigned int *Ls, int *center, unsigned int *Down_smp, float C);

#else
int pcc_highlevel (float **y, _Complex float **x1, _Complex float **x2, int N, unsigned int Tr, float v, int Lag1, int Lag2, sem_t *anok);
int pcc1_highlevel (float **y, _Complex float **x1, _Complex float **x2, int N, unsigned int Tr, int Lag1, int Lag2, sem_t *anok);
void pcc1_highlevel2 (float **y, float **x1, float **x2, int N, unsigned int Tr, int Lag1, int Lag2);
int cuda_tspcc2_set_freq (float **y0, float **x01, float **x02, unsigned int N, unsigned int Tr, unsigned int Nz, const int Lag1, const int Lag2, 
	unsigned int S, double *scale, _Complex double **wc, unsigned int *Ls, int *center, unsigned int *Down_smp, float C);
#endif

#endif
