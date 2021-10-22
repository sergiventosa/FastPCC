#ifndef CCS_CUDA_H
#define CCS_CUDA_H

#include <complex.h>
#include <semaphore.h>

#ifdef __cplusplus
extern "C" int pcc1_highlevel (float **y, _Complex float **x1, _Complex float **x2, int N, unsigned int Tr, int Lag1, int Lag2, sem_t *anok);
extern "C" void pcc1_highlevel2 (float **y, float **x1, float **x2, int N, unsigned int Tr, int Lag1, int Lag2);

#else
int pcc1_highlevel (float **y, _Complex float **x1, _Complex float **x2, int N, unsigned int Tr, int Lag1, int Lag2, sem_t *anok);
void pcc1_highlevel2 (float **y, float **x1, float **x2, int N, unsigned int Tr, int Lag1, int Lag2);
#endif

#endif
