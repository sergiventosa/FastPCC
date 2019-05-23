#include <complex.h>
#include <fftw3.h>
#include <stdio.h>
#include <math.h>
#include <fcntl.h>
#include <semaphore.h>
#include <cuda_runtime.h>
#include "ccs_cuda.h"

const int BLOCK_SIZE=256;

// Device code
__global__ void GPU_AmpNormf(float2 *x, unsigned int N) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;
	
	float fa1, fa2, fa3;
	
	if (n<N) {
		fa1 = x[n].x;
		fa2 = x[n].y;
		fa3 = fa1*fa1+fa2*fa2;
		if (fa3 != 0) {
			fa3 = 1/sqrtf(fa3);
			x[n].x *= fa3;
			x[n].y *= fa3;
		} else {
			x[n].x = 0;
			x[n].y = 0;
		}
	}
}

__global__ void PCC1_lowlevel (float *x1, float *x2, float *y, int N, const int L, const int l1, const int Lag1) {
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	
	float fa1, fa2, fa3, fa4, fa5;
	float xa[2], xb[2];
	int lag, n;
	
	if (l >= l1 && l < L) {
		lag = Lag1 + l;
		if (lag < 0) { x2 -= 2*lag;  N += lag; }
		else         { x1 += 2*lag;  N -= lag; }
		
		fa5 = 0;
		for (n=0; n<2*N; n+=2) {
			xa[0] = x1[n]; xa[1] = x1[n+1];
			xb[0] = x2[n]; xb[1] = x2[n+1];
			fa1 = xa[0] + xb[0];
			fa2 = xa[1] + xb[1];
			fa3 = xa[0] - xb[0];
			fa4 = xa[1] - xb[1];
			fa5 += sqrtf(fa1*fa1+fa2*fa2) - sqrtf(fa3*fa3+fa4*fa4);
		}
		y[l] = fa5/(2*N);
	}
}

__global__ void PCC1_lowlevel2 (const float2 *x1, const float2 *x2, float *y, int N, const int L, const int l1, const int Lag1) {
	int l = blockDim.x * blockIdx.x + threadIdx.x; /* One thread per lag */
	int ls = threadIdx.x;
	__shared__ float2 x1s[2*BLOCK_SIZE]; /* x1 & x2 are in fact float-complex types */
	__shared__ float2 x2s[BLOCK_SIZE];

	float fa1, fa2, fa3, fa4, fa5;
	float2 *xa, *xb;
	int n, nb;
	
	/* The smallest domain is one block */
	fa5 = 0;
	for (nb=0; nb<N; nb+=BLOCK_SIZE) {
		/*** Copy data from the device to the shared memory ***/ 
		n = nb + ls; /* One sample of x1[n] */
		if (n < N) {
			x2s[ls].x = x2[n].x;
			x2s[ls].y = x2[n].y;
		} else {
			x2s[ls].x = 0;
			x2s[ls].y = 0;
		}
		
		n = nb + Lag1 + l; /* Sample from x2 aligned to the sample of x1 read above when threadIdx.x = 0. */
		if (n >= 0 && n < N) {
			x1s[ls].x = x1[n].x;
			x1s[ls].y = x1[n].y;
		} else {
			x1s[ls].x = 0;
			x1s[ls].y = 0;
		}
		
		n  += BLOCK_SIZE;
		ls += BLOCK_SIZE;
		if (n >= 0 && n < N) {
			x1s[ls].x = x1[n].x;
			x1s[ls].y = x1[n].y;
		} else {
			x1s[ls].x = 0;
			x1s[ls].y = 0;
		}
		
		__syncthreads();
		
		/*** Partial PCC1 ***/
		ls = threadIdx.x;
		if (l >= l1 && l < L) {
			xb = x1s + ls;
			xa = x2s;
			for (n=0; n<BLOCK_SIZE; n++) {
				fa1 = xa[n].x + xb[n].x;
				fa2 = xa[n].y + xb[n].y;
				fa3 = xa[n].x - xb[n].x;
				fa4 = xa[n].y - xb[n].y;
				fa5 += sqrtf(fa1*fa1+fa2*fa2) - sqrtf(fa3*fa3+fa4*fa4);
			}
		}
		__syncthreads();
	}
	y[l] = fa5/(2*(N-abs(Lag1 + l)));
}

// Host code
#if 0
void pcc1_highlevel (double ** const y, _Complex float ** const x1, _Complex float ** const x2, const int N, const int Tr, const int Lag1, const int Lag2, const unsigned int *trid) {
	float *h_y;
	float2 *d_xan1, *d_xan2;
	float *d_y;
	int L=Lag2-Lag1+1;
	size_t szx, szy;
	unsigned int tr, id, l, l1;
	int threadsPerBlock = BLOCK_SIZE, blocksPerGrid;
	sem_t *anok;
	
	if (Lag2 > N) L -= (Lag2-N);
	l1 = (Lag1 >= -N) ? 0 : -(Lag1+N);

	if (SEM_FAILED == (anok = sem_open("/analytic", O_CREAT, 0600, 0) )) {
		printf("pcc1_highlevel: sem_open fail\n");
		return;
	}
	/* Convert data to float */
	szx = N*sizeof(float2);
	szy = L*sizeof(float);
	
	/* Allocate vectors in device & host memories */
	cudaMallocHost(&h_y, szy);
	cudaMalloc(&d_xan1, szx);
	cudaMalloc(&d_xan2, szx);
	cudaMalloc(&d_y,    szy);
	
	anok = sem_open("/analytic", O_CREAT, 0600, 0);
	if (anok != SEM_FAILED) {
		for (tr=0; tr<Tr; tr++) {
			sem_wait(anok);
			id = trid[tr];
			
			cudaMemcpy(d_xan1, (float *)x1[id], szx, cudaMemcpyHostToDevice);
			cudaMemcpy(d_xan2, (float *)x2[id], szx, cudaMemcpyHostToDevice);
			
			// Invoke the kernel GPU_AmpNormf
			blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
			GPU_AmpNormf<<<blocksPerGrid, threadsPerBlock>>>(d_xan1, N);
			GPU_AmpNormf<<<blocksPerGrid, threadsPerBlock>>>(d_xan2, N);
			
			// Invoke the kernel PCC1_lowlevel
			blocksPerGrid = (L-l1 + threadsPerBlock - 1) / threadsPerBlock;
			PCC1_lowlevel2<<<blocksPerGrid, threadsPerBlock>>>(d_xan1, d_xan2, d_y, N, L, l1, Lag1);
			
			/* Copy result back */
			cudaMemcpy(h_y, d_y, szy, cudaMemcpyDeviceToHost);
			
			/* Convert results to double */
			for (l=l1; l<L; l++) y[id][l] = h_y[l];
		}
		sem_close(anok);
	}
	cudaFree(d_xan1);
	cudaFree(d_xan2);
	cudaFree(d_y);
	cudaFreeHost(h_y);
}
#else

void pcc1_highlevel (double **y, _Complex float **x1, _Complex float **x2, int N, int Tr, int Lag1, int Lag2, const unsigned int *trid) {
	float *h_y;
	float2 *d_xan1, *d_xan2;
	float *d_y;
	int L=Lag2-Lag1+1;
	size_t szx, szy;
	unsigned int tr, id, l, l1, n, m;
	int threadsPerBlock = BLOCK_SIZE, blocksPerGrid1, blocksPerGrid2;
	cudaStream_t stream[16];
	sem_t *anok;
	
	if (Lag2 > N) L -= (Lag2-N);
	l1 = (Lag1 >= -N) ? 0 : -(Lag1+N);

	/* Convert data to float */
	szx = N*sizeof(float2);
	szy = L*sizeof(float);
	
	/* Allocate vectors in device & host memories */
	cudaMallocHost(&h_y, Tr*szy);
	cudaMalloc(&d_xan1, 16*szx);
	cudaMalloc(&d_xan2, 16*szx);
	cudaMalloc(&d_y,    16*szy);
	
	blocksPerGrid1 = (N + threadsPerBlock - 1) / threadsPerBlock;
	blocksPerGrid2 = (L-l1 + threadsPerBlock - 1) / threadsPerBlock;
	for (m=0; m<16; m++) cudaStreamCreate(&stream[m]);
	anok = sem_open("/analytic", O_CREAT, 0600, 0);
	if (anok != SEM_FAILED) {
		for (n=0; n<(Tr+15)/16; n++) {
			for (m=0; m<16; m++) {
				tr = 16*n+m;
				if (tr < Tr) {
					sem_wait(anok);
					id = trid[tr];
					
					cudaMemcpy(d_xan1 + m*N, (float *)x1[id], szx, cudaMemcpyHostToDevice);
					cudaMemcpy(d_xan2 + m*N, (float *)x2[id], szx, cudaMemcpyHostToDevice);
					
					// Invoke the kernel GPU_AmpNormf
					GPU_AmpNormf<<<blocksPerGrid1, threadsPerBlock, 0, stream[m]>>>(d_xan1 + m*N, N);
					GPU_AmpNormf<<<blocksPerGrid1, threadsPerBlock, 0, stream[m]>>>(d_xan2 + m*N, N);
					
					// Invoke the kernel PCC1_lowlevel
					PCC1_lowlevel2<<<blocksPerGrid2, threadsPerBlock, 0, stream[m]>>>(d_xan1 + m*N, d_xan2 + m*N, d_y + m*L, N, L, l1, Lag1);
					
					/* Copy result back */
					cudaMemcpyAsync(h_y + id*L, d_y + m*L, szy, cudaMemcpyDeviceToHost, stream[m]);
				}
			}
		}
		sem_close(anok);
	}
	cudaDeviceSynchronize();
	
	for (m=0; m<16; m++) cudaStreamDestroy(stream[m]);
	
	/* Convert results to double */
	for (tr=0; tr<Tr; tr++)
		for (l=l1; l<L; l++) y[tr][l] = h_y[tr*L + l];
	
	cudaFree(d_xan1);
	cudaFree(d_xan2);
	cudaFree(d_y);
	cudaFreeHost(h_y);
}
#endif
