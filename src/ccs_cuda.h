#ifndef CCS_CUDA_H
#define CCS_CUDA_H

#include <complex.h>
#include <semaphore.h>

#ifdef __cplusplus
extern "C" void pcc1_highlevel (double **y, _Complex float **x1, _Complex float **x2, int N, int Tr, int Lag1, int Lag2, sem_t *anok);
#else
void pcc1_highlevel (double **y, _Complex float **x1, _Complex float **x2, int N, int Tr, int Lag1, int Lag2, sem_t *anok);
#endif

#endif
