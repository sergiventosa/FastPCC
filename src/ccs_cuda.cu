#include <complex.h>
#include <fftw3.h>
#include <stdio.h>
#include <math.h>
#include <semaphore.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "ccs_cuda.h"

#define BLOCK_SIZE 256

// Device code
__global__ void GPU_analytic (cufftComplex *y, unsigned int N) {
	unsigned int n = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tr = blockIdx.y;
	float da1 = 1/(float)N;
	unsigned int nh;
	
	nh = N/2+1;
	if (n > nh && n < N) {
		y[n+N*tr].x = 0;
		y[n+N*tr].y = 0;
	} else if (n > 0 && !(N&1)) {
		y[n+N*tr].x *= 2*da1;
		y[n+N*tr].y *= 2*da1;
	} else {
		y[n+N*tr].x *= da1;
		y[n+N*tr].y *= da1;
	}
}

__global__ void GPU_xcorr (cufftComplex *out, cufftComplex *xa1, cufftComplex *xa2, float da1, unsigned int N) {
	int n = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (n<N) {
		out[n].x = da1 * (xa1[n].x * xa2[n].x + xa1[n].y * xa2[n].y);
		out[n].y = da1 * (xa1[n].y * xa2[n].x - xa1[n].x * xa2[n].y);
	}
}

__global__ void GPU_copy_and_normalized_lags (float *y, float2 *x, unsigned int Nz, float C, int lag, int Lags) {
	unsigned int n = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (n < -lag)      y[n] = C * x[n+Nz+lag].x;
	else if (n < Lags) y[n] = C * x[n+lag].x;
}

__global__ void GPU_F2C (float2 *y, float *x, unsigned int N) {
	unsigned int n = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tr = blockIdx.y;
	
	if (n<N) {
		y[n+N*tr].x = x[n+N*tr];
		y[n+N*tr].y = 0.;
	}
}

int GPU_AnalyticSignal (cufftComplex *y, cufftReal *x, unsigned int N, cufftHandle *pin, cufftHandle *pout) {
	int threadsPerBlock = BLOCK_SIZE, blocksPerGrid;

	blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	cudaMemset(y, 0, N*sizeof(cufftComplex));
	
	/* out = FFT(in) */
	if (cufftExecR2C(*pin, x, y) != CUFFT_SUCCESS) { fprintf(stderr, "cufftExecR2C failed\n"); return 1; }

	/* Make it analytic ( out(w<0) = 0, out(w>0) *= 2, out(0) & out(Nyquist) no change). */
	GPU_analytic<<<blocksPerGrid, threadsPerBlock>>>(y, N);

	/* in = IFFT(out) */
	if (cufftExecC2C(*pout, y, y, CUFFT_INVERSE) != CUFFT_SUCCESS) { fprintf(stderr, "cufftExecC2C failed\n"); return 1; }
	
	return 0;
}

/* Sum reduction of the shared float array with the length of BLOCK_SIZE. */
template <unsigned int block_size> __device__ void SumReduction (volatile float *x, unsigned int tid) {
	if (block_size >= 1024) { if (tid < 512) x[tid] += x[tid + 512]; __syncthreads(); }
	if (block_size >= 512)  { if (tid < 256) x[tid] += x[tid + 256]; __syncthreads(); }
	if (block_size >= 256)  { if (tid < 128) x[tid] += x[tid + 128]; __syncthreads(); }
	if (block_size >= 128)  { if (tid < 64)  x[tid] += x[tid +  64]; __syncthreads(); }
	
	if (tid < 32) x[tid] += x[tid + 32];
	if (tid < 16) x[tid] += x[tid + 16];
	if (tid < 8)  x[tid] += x[tid +  8];
	if (tid < 4)  x[tid] += x[tid +  4];
	if (tid < 2)  x[tid] += x[tid +  2];
	if (tid == 0) x[0] += x[1];
}


/* Max reduction of the shared float array with the length of BLOCK_SIZE. */
template <unsigned int block_size> __device__ void MaxReduction (volatile float *x, unsigned int tid) {
	if (block_size >= 1024) { if (tid < 512) x[tid] = fmaxf(x[tid], x[tid + 512]); __syncthreads(); }
	if (block_size >= 512)  { if (tid < 256) x[tid] = fmaxf(x[tid], x[tid + 256]); __syncthreads(); }
	if (block_size >= 256)  { if (tid < 128) x[tid] = fmaxf(x[tid], x[tid + 128]); __syncthreads(); }
	if (block_size >= 128)  { if (tid < 64)  x[tid] = fmaxf(x[tid], x[tid +  64]); __syncthreads(); }
	
	if (tid < 32) x[tid] = fmaxf(x[tid], x[tid + 32]);
	if (tid < 16) x[tid] = fmaxf(x[tid], x[tid + 16]);
	if (tid < 8)  x[tid] = fmaxf(x[tid], x[tid +  8]);
	if (tid < 4)  x[tid] = fmaxf(x[tid], x[tid +  4]);
	if (tid < 2)  x[tid] = fmaxf(x[tid], x[tid +  2]);
	if (tid == 0) x[0] = fmaxf(x[tid], x[1]);
}

__global__ void GPU_norm_ln(float *y, float2 *x, unsigned int N) {
	__shared__ float a[BLOCK_SIZE];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = 2*blockIdx.x*blockDim.x + tid;
	unsigned int j;
	
	a[tid] = 0;
	if (i<N) a[tid]  = x[i].x*x[i].x + x[i].y*x[i].y; 
	__syncthreads();
	
	j = i + blockDim.x;
	if (j<N) a[tid] += x[j].x*x[j].x + x[j].y*x[j].y;
	__syncthreads();
	
	SumReduction <BLOCK_SIZE> (a, tid);
	
	if (tid == 0) y[blockIdx.x] = a[0]; /* Here each block has the squared norm of its elements */
}

__global__ void GPU_mabs_reduction(float *x, unsigned int M, unsigned int N) {
	__shared__ float a[BLOCK_SIZE];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = 2*blockIdx.x*blockDim.x + tid;
	
	a[tid] = 0;
	if (i<M) a[tid] = x[i];
	__syncthreads();
	
	if (i + blockDim.x<M) a[tid] += x[i + blockDim.x];
	__syncthreads();
	
	SumReduction <BLOCK_SIZE> (a, tid);
	
	if (tid == 0) {
		if (blockDim.x == 1) x[0] = sqrtf(a[0])/N;
		x[blockIdx.x] = a[0];
	}
}

__global__ void GPU_max_ln(float *y, float2 *x, unsigned int N) {
	__shared__ float a[BLOCK_SIZE];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = 2*blockIdx.x*blockDim.x + tid;
	unsigned int j;
	
	a[tid] = 0;
	if (i<N) a[tid] = x[i].x*x[i].x + x[i].y*x[i].y; 
	__syncthreads();
	
	j = i + blockDim.x;
	if (j<N) a[tid] = fmaxf(a[tid], x[j].x*x[j].x + x[j].y*x[j].y);
	__syncthreads();
	
	MaxReduction <BLOCK_SIZE> (a, tid);
	
	if (tid == 0) y[blockIdx.x] = a[0]; /* Here each block has the squared norm of its elements */
}

__global__ void GPU_max_reduction(float *x, unsigned int M, unsigned int N) {
	__shared__ float a[BLOCK_SIZE];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = 2*blockIdx.x*blockDim.x + tid;
	
	a[tid] = 0;
	if (i<M) a[tid] = x[i];
	__syncthreads();
	
	if (i + blockDim.x<M) a[tid] = fmaxf(a[tid], x[i + blockDim.x]);
	__syncthreads();
	
	MaxReduction <BLOCK_SIZE> (a, tid);
	
	if (tid == 0) {
		if (blockDim.x == 1) x[0] = sqrtf(a[0]);
		x[blockIdx.x] = a[0];
	};
}


__global__ void GPU_AmpNormf2(float2 *x, unsigned int N, float *mabs) {
	unsigned int n = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tr = blockIdx.y;
	
	float e, fa1, fa2, fa3;
	
	if (n<N) {
		e = 1e-6*mabs[0];
		fa1 = x[n+N*tr].x;
		fa2 = x[n+N*tr].y;
		fa3 = fa1*fa1+fa2*fa2;
		if (fa3 > e) {
			fa3 = 1/sqrtf(fa3);
			x[n+N*tr].x = fa1*fa3;
			x[n+N*tr].y = fa2*fa3;
		} else {
			x[n+N*tr].x = 0;
			x[n+N*tr].y = 0;
		}
	}
}

__global__ void GPU_AmpNormf(float2 *x, unsigned int N, float mabs) {
	unsigned int n = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tr = blockIdx.y;
	
	float e, fa1, fa2, fa3;
	
	if (n<N) {
		e = 1e-6*mabs;
		fa1 = x[n+N*tr].x;
		fa2 = x[n+N*tr].y;
		fa3 = fa1*fa1+fa2*fa2;
		if (fa3 > e) {
			fa3 = 1/sqrtf(fa3);
			x[n+N*tr].x = fa1*fa3;
			x[n+N*tr].y = fa2*fa3;
		} else {
			x[n+N*tr].x = 0;
			x[n+N*tr].y = 0;
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
	if (l >= l1 && l < L)
		y[l] = fa5/(2*(N-abs(Lag1 + l)));
}

__global__ void PCC1_lowlevel3 (const float2 *x1, const float2 *x2, float *y, int N, const int L, const int l1, const int Lag1) {
	int l = blockDim.x * blockIdx.x + threadIdx.x; /* One thread per lag */
	int ls = threadIdx.x;
	int cmp1 = blockIdx.y;
	int cmp2 = blockIdx.z;
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
			x2s[ls].x = x2[n+N*cmp2].x;
			x2s[ls].y = x2[n+N*cmp2].y;
		} else {
			x2s[ls].x = 0;
			x2s[ls].y = 0;
		}
		
		n = nb + Lag1 + l; /* Sample from x2 aligned to the sample of x1 read above when threadIdx.x = 0. */
		if (n >= 0 && n < N) {
			x1s[ls].x = x1[n+N*cmp1].x;
			x1s[ls].y = x1[n+N*cmp1].y;
		} else {
			x1s[ls].x = 0;
			x1s[ls].y = 0;
		}
		
		n  += BLOCK_SIZE;
		ls += BLOCK_SIZE;
		if (n >= 0 && n < N) {
			x1s[ls].x = x1[n+N*cmp1].x;
			x1s[ls].y = x1[n+N*cmp1].y;
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
	if (l >= l1 && l < L)
		y[l+L*(3*cmp1+cmp2)] = fa5/(2*(N-abs(Lag1 + l)));
}

__global__ void PCC_lowlevel3 (const float2 *x1, const float2 *x2, float *y, int N, float v, const int L, const int l1, const int Lag1) {
	int l = blockDim.x * blockIdx.x + threadIdx.x; /* One thread per lag */
	int ls = threadIdx.x;
	int cmp1 = blockIdx.y;
	int cmp2 = blockIdx.z;
	__shared__ float2 x1s[2*BLOCK_SIZE]; /* x1 & x2 are in fact float-complex types */
	__shared__ float2 x2s[BLOCK_SIZE];

	float fa1, fa2, fa3, fa4, fa5;
	float2 *xa, *xb;
	int n, nb;
	
	v *= 0.5;
	
	/* The smallest domain is one block */
	fa5 = 0;
	for (nb=0; nb<N; nb+=BLOCK_SIZE) {
		/*** Copy data from the device to the shared memory ***/ 
		n = nb + ls; /* One sample of x1[n] */
		if (n < N) {
			x2s[ls].x = x2[n+N*cmp2].x;
			x2s[ls].y = x2[n+N*cmp2].y;
		} else {
			x2s[ls].x = 0;
			x2s[ls].y = 0;
		}
		
		n = nb + Lag1 + l; /* Sample from x2 aligned to the sample of x1 read above when threadIdx.x = 0. */
		if (n >= 0 && n < N) {
			x1s[ls].x = x1[n+N*cmp1].x;
			x1s[ls].y = x1[n+N*cmp1].y;
		} else {
			x1s[ls].x = 0;
			x1s[ls].y = 0;
		}
		
		n  += BLOCK_SIZE;
		ls += BLOCK_SIZE;
		if (n >= 0 && n < N) {
			x1s[ls].x = x1[n+N*cmp1].x;
			x1s[ls].y = x1[n+N*cmp1].y;
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
				fa5 += powf(fa1*fa1+fa2*fa2, v) - powf(fa3*fa3+fa4*fa4, v);
			}
		}
		__syncthreads();
	}
	if (l >= l1 && l < L)
		y[l+L*(3*cmp1+cmp2)] = fa5/(2*(N-abs(Lag1 + l)));
}

// Host code

void cuda_AmpNormf(float2 *x, unsigned int N) {
	float eps, *norm_ln;
	int M, nth = BLOCK_SIZE, nbl;
	// loat *tmp;
	// int m;
	
	/* Compute reguralitzation as 1e-6 part of the mean modulus of x. */
	nbl = (N + 2*nth - 1) / (2*nth);                     /* One thread per two elements on x. */
	cudaMalloc(&norm_ln, nbl*sizeof(float));             /* Buffer for partiar norms.         */
	// cudaMallocHost(&tmp, nbl*sizeof(float));             /* Buffer for partiar norms.         */
	
	GPU_norm_ln<<<nbl, nth>>>(norm_ln, x, N);  /* First iteration, squared modulus of each block */
	// cudaMemcpy(tmp, norm_ln, nbl*sizeof(float), cudaMemcpyDeviceToHost);
	
	/* printf("\nM=%d\n", nbl);
	for (m=0; m<nbl; m++) printf("%g ", tmp[m]);
	printf("\n"); */
	
	while (nbl > 1) {
		M = nbl;
		nbl = (M + 2*nth - 1) / (2*nth);
		// printf("nbl=%d\n", nbl);
		GPU_mabs_reduction<<<nbl, nth>>>(norm_ln, M, N); /* Iterate until a single block remains. */
	}
	cudaMemcpy(&eps, norm_ln, sizeof(float), cudaMemcpyDeviceToHost);
	
	// printf("N=%d Norm^2=%g\n", N, eps);
	cudaFree(norm_ln);
	// cudaFreeHost(tmp);
	
	/* Amplitude Normalization */
	nbl = (N + nth - 1) / nth;
	// printf("eps = %g\n", eps);
	GPU_AmpNormf<<<nbl, nth>>>(x, N, eps);
}

void cuda_AmpNormf2(float2 *x, unsigned int N, float *norm_ln, cudaStream_t *stream) {
	int M, nth = BLOCK_SIZE, nbl;
	
	/* Compute reguralitzation as 1e-6 part of the mean modulus of x. */
	nbl = (N + 2*nth - 1) / (2*nth);                     /* One thread per two elements on x. */
	if (stream) GPU_norm_ln<<<nbl, nth, 0, stream[0]>>>(norm_ln, x, N);  /* First iteration, squared modulus of each block */
	else		GPU_norm_ln<<<nbl, nth>>>(norm_ln, x, N); 
	while (nbl > 1) {
		M = nbl;
		nbl = (M + 2*nth - 1) / (2*nth);
		if (stream) GPU_mabs_reduction<<<nbl, nth, 0, stream[0]>>>(norm_ln, M, N); /* Iterate until a single block remains. */
		else        GPU_mabs_reduction<<<nbl, nth>>>(norm_ln, M, N); /* Iterate until a single block remains. */
	}
	
	/* Amplitude Normalization */
	nbl = (N + nth - 1) / nth;
	if (stream)  GPU_AmpNormf2<<<nbl, nth, 0, stream[0]>>>(x, N, norm_ln);
	else         GPU_AmpNormf2<<<nbl, nth>>>(x, N, norm_ln);
}

void cuda_AmpNormf3(float2 *x, unsigned int N, float *max_ln, cudaStream_t *stream) {
	int M, nth = BLOCK_SIZE, nbl;
	
	/* Compute reguralitzation as 1e-6 part of the max modulus of x. */
	nbl = (N + 2*nth - 1) / (2*nth);                     /* One thread per two elements on x. */
	if (stream) GPU_max_ln<<<nbl, nth, 0, stream[0]>>>(max_ln, x, N);  /* First iteration, squared modulus of each block */
	else		GPU_max_ln<<<nbl, nth>>>(max_ln, x, N); 
	while (nbl > 1) {
		M = nbl;
		nbl = (M + 2*nth - 1) / (2*nth);
		if (stream) GPU_max_reduction<<<nbl, nth, 0, stream[0]>>>(max_ln, M, N); /* Iterate until a single block remains. */
		else        GPU_max_reduction<<<nbl, nth>>>(max_ln, M, N); /* Iterate until a single block remains. */
	}
	
	/* Amplitude Normalization */
	nbl = (N + nth - 1) / nth;
	if (stream)  GPU_AmpNormf2<<<nbl, nth, 0, stream[0]>>>(x, N, max_ln);
	else         GPU_AmpNormf2<<<nbl, nth>>>(x, N, max_ln);
}

int pcc_highlevel_error (const char *str, cudaError_t cudaerr) {
	printf ("%s :%s\n", str, cudaGetErrorString(cudaerr));
	cudaDeviceReset();
	return -1;
}

int pcc_highlevel (float **y, _Complex float **x1, _Complex float **x2, int N, unsigned int Tr, float v, int Lag1, int Lag2, sem_t *anok) {
	float *h_y;
	float2 *d_xan1, *d_xan2;
	float *d_y;
	size_t szx, szy;
	unsigned int tr, l, l1, n, m, L=(unsigned)abs(Lag2-Lag1+1);
	int threadsPerBlock = BLOCK_SIZE, blocksPerGrid1, blocksPerGrid2;
	size_t available=0, total=0;
	cudaStream_t stream[16];
	cudaError_t cudaerr;
	
	if (Lag2 > N) L -= (Lag2-N);
	l1 = (Lag1 >= -N) ? 0 : -(Lag1+N);

	/* Convert data to float */
	szx = N*sizeof(float2);
	szy = L*sizeof(float);
	
	cudaerr = cudaMemGetInfo(&available, &total);
	if (cudaerr != cudaSuccess) 
		{ printf ("Error getting memory info (%s)\n", cudaGetErrorString(cudaerr)); return -1; }
	
	/* Allocate vectors in device & host memories */
	cudaerr = cudaMallocHost(&h_y, Tr*szy);
		if (cudaerr != cudaSuccess) return pcc_highlevel_error("Error allocating pinned host memory h_y", cudaerr);
	cudaMalloc(&d_xan1, 16*szx);
	cudaMalloc(&d_xan2, 16*szx);
	cudaMalloc(&d_y,    16*szy);
	
	blocksPerGrid1 = (N + threadsPerBlock - 1) / threadsPerBlock;
	blocksPerGrid2 = (L-l1 + threadsPerBlock - 1) / threadsPerBlock;
	for (m=0; m<16; m++) cudaStreamCreate(&stream[m]);
	for (n=0; n<((unsigned)Tr+15)/16; n++) {
		for (m=0; m<16; m++) {
			tr = 16*n+m;
			if (tr < (unsigned)Tr) {
				sem_wait(&anok[tr]);
				
				// Copy data to the GPU memory
				cudaMemcpyAsync(d_xan1 + m*N, (float *)x1[tr], szx, cudaMemcpyHostToDevice, stream[m]);
				cudaMemcpyAsync(d_xan2 + m*N, (float *)x2[tr], szx, cudaMemcpyHostToDevice, stream[m]);
				
				// Invoke the kernel GPU_AmpNormf
				GPU_AmpNormf<<<blocksPerGrid1, threadsPerBlock, 0, stream[m]>>>(d_xan1 + m*N, N, 0);
				GPU_AmpNormf<<<blocksPerGrid1, threadsPerBlock, 0, stream[m]>>>(d_xan2 + m*N, N, 0);
				
				// Invoke the kernel PCC1_lowlevel
				PCC_lowlevel3<<<blocksPerGrid2, threadsPerBlock, 0, stream[m]>>>(d_xan2 + m*N, d_xan1 + m*N, d_y + m*L, N, v, L, l1, Lag1);
				
				/* Copy result back */
				cudaMemcpyAsync(h_y + tr*L, d_y + m*L, szy, cudaMemcpyDeviceToHost, stream[m]);
			}
		}
	}
	cudaDeviceSynchronize();
	
	for (m=0; m<16; m++) cudaStreamDestroy(stream[m]);
	
	for (tr=0; tr<Tr; tr++)
		for (l=l1; l<L; l++) y[tr][l] = h_y[tr*L + l];
	
	cudaFree(d_xan1);
	cudaFree(d_xan2);
	cudaFree(d_y);
	cudaFreeHost(h_y);
	
	return 0;
}

#if 0
int pcc1_highlevel (float **y, _Complex float **x1, _Complex float **x2, int N, unsigned int Tr, int Lag1, int Lag2, sem_t *anok) {
	float *h_y;
	float2 *d_xan1, *d_xan2;
	float *d_y;
	size_t szx, szy;
	unsigned int tr, l, l1, L=(unsigned)abs(Lag2-Lag1+1);
	int threadsPerBlock = BLOCK_SIZE, blocksPerGrid;
	
	if (Lag2 > N) L -= (Lag2-N);
	l1 = (Lag1 >= -N) ? 0 : -(Lag1+N);

	/* Convert data to float */
	szx = N*sizeof(float2);
	szy = L*sizeof(float);
	
	/* Allocate vectors in device & host memories */
	cudaMallocHost(&h_y, szy);
	cudaMalloc(&d_xan1, szx);
	cudaMalloc(&d_xan2, szx);
	cudaMalloc(&d_y,    szy);
	
	for (tr=0; tr<Tr; tr++) {
		sem_wait(&anok[tr]);
		
		cudaMemcpy(d_xan1, (float *)x1[tr], szx, cudaMemcpyHostToDevice);
		cudaMemcpy(d_xan2, (float *)x2[tr], szx, cudaMemcpyHostToDevice);
		
		// Invoke the kernel GPU_AmpNormf including regurization.
		cuda_AmpNormf(d_xan1, N);
		cuda_AmpNormf(d_xan2, N);
		
		// Invoke the kernel PCC1_lowlevel
		blocksPerGrid = (L-l1 + threadsPerBlock - 1) / threadsPerBlock;
		PCC1_lowlevel2<<<blocksPerGrid, threadsPerBlock>>>(d_xan2, d_xan1, d_y, N, L, l1, Lag1);
		
		/* Copy result back */
		cudaMemcpy(h_y, d_y, szy, cudaMemcpyDeviceToHost);
		for (l=l1; l<L; l++) y[tr][l] = h_y[l];
	}
	
	cudaFree(d_xan1);
	cudaFree(d_xan2);
	cudaFree(d_y);
	cudaFreeHost(h_y);
	
	return 0;
}
#else
int pcc1_highlevel (float **y, _Complex float **x1, _Complex float **x2, int N, unsigned int Tr, int Lag1, int Lag2, sem_t *anok) {
	float *h_y;
	float2 *d_xan1, *d_xan2;
	float *d_y;
	size_t szx, szy;
	unsigned int tr, l, l1, n, m, L=(unsigned)abs(Lag2-Lag1+1);
	int threadsPerBlock = BLOCK_SIZE, blocksPerGrid1, blocksPerGrid2;
	size_t available=0, total=0;
	cudaStream_t stream[16];
	cudaError_t cudaerr;
	
	if (Lag2 > N) L -= (Lag2-N);
	l1 = (Lag1 >= -N) ? 0 : -(Lag1+N);

	/* Convert data to float */
	szx = N*sizeof(float2);
	szy = L*sizeof(float);
	
	cudaerr = cudaMemGetInfo(&available, &total);
	if (cudaerr != cudaSuccess) 
		{ printf ("Error getting memory info (%s)\n", cudaGetErrorString(cudaerr)); return -1; }
	
	/* Allocate vectors in device & host memories */
	cudaerr = cudaMallocHost(&h_y, Tr*szy);
		if (cudaerr != cudaSuccess) return pcc_highlevel_error("Error allocating pinned host memory h_y", cudaerr);
	cudaMalloc(&d_xan1, 16*szx);
	cudaMalloc(&d_xan2, 16*szx);
	cudaMalloc(&d_y,    16*szy);
	
	blocksPerGrid1 = (N + threadsPerBlock - 1) / threadsPerBlock;
	blocksPerGrid2 = (L-l1 + threadsPerBlock - 1) / threadsPerBlock;
	for (m=0; m<16; m++) cudaStreamCreate(&stream[m]);
	for (n=0; n<((unsigned)Tr+15)/16; n++) {
		for (m=0; m<16; m++) {
			tr = 16*n+m;
			if (tr < (unsigned)Tr) {
				sem_wait(&anok[tr]);
				
				// Copy data to the GPU memory
				cudaMemcpyAsync(d_xan1 + m*N, (float *)x1[tr], szx, cudaMemcpyHostToDevice, stream[m]);
				cudaMemcpyAsync(d_xan2 + m*N, (float *)x2[tr], szx, cudaMemcpyHostToDevice, stream[m]);
				
				// Invoke the kernel GPU_AmpNormf
				GPU_AmpNormf<<<blocksPerGrid1, threadsPerBlock, 0, stream[m]>>>(d_xan1 + m*N, N, 0);
				GPU_AmpNormf<<<blocksPerGrid1, threadsPerBlock, 0, stream[m]>>>(d_xan2 + m*N, N, 0);
				
				// Invoke the kernel PCC1_lowlevel
				PCC1_lowlevel2<<<blocksPerGrid2, threadsPerBlock, 0, stream[m]>>>(d_xan2 + m*N, d_xan1 + m*N, d_y + m*L, N, L, l1, Lag1);
				
				/* Copy result back */
				cudaMemcpyAsync(h_y + tr*L, d_y + m*L, szy, cudaMemcpyDeviceToHost, stream[m]);
			}
		}
	}
	cudaDeviceSynchronize();
	
	for (m=0; m<16; m++) cudaStreamDestroy(stream[m]);
	
	for (tr=0; tr<Tr; tr++)
		for (l=l1; l<L; l++) y[tr][l] = h_y[tr*L + l];
	
	cudaFree(d_xan1);
	cudaFree(d_xan2);
	cudaFree(d_y);
	cudaFreeHost(h_y);
	
	return 0;
}
#endif

#if 0
void pcc1_highlevel2 (float **y, float **x1, float **x2, int N, unsigned int Tr, int Lag1, int Lag2) {
	cufftHandle pin, pout;
	cufftReal *d_x=NULL;
	float *h_x, *h_y, *d_y, *mem;
	float2 *d_xan1, *d_xan2;
	size_t szxr, szxc, szy;
	unsigned int tr, l, l1, L=(unsigned)abs(Lag2-Lag1+1);
	int threadsPerBlock = BLOCK_SIZE, blocksPerGrid;
	
	blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	
	if (Lag2 > N) L -= (Lag2-N);
	l1 = (Lag1 >= -N) ? 0 : -(Lag1+N);

	/* Convert data to float */
	szxr = N*sizeof(float);
	szxc = N*sizeof(float2);
	szy = L*sizeof(float);
	
	cudaMallocHost(&h_x, szxr);
	cudaMallocHost(&h_y, szy);
	
	/* Allocate vectors in device memory */
	cudaMalloc(&d_x,    szxr);
	cudaMalloc(&d_xan1, szxc);
	cudaMalloc(&d_xan2, szxc);
	cudaMalloc(&d_y,    szy);
	cudaMalloc(&mem, blocksPerGrid*sizeof(float));
	
	/* Make the plans */
	if (cufftPlan1d(&pin,  N, CUFFT_R2C, 1) != CUFFT_SUCCESS) { fprintf(stderr, "CUFFPlan1d failed\n"); return; }
	if (cufftPlan1d(&pout, N, CUFFT_C2C, 1) != CUFFT_SUCCESS) { fprintf(stderr, "CUFFPlan1d failed\n"); return; }
	
	for (tr=0; tr<Tr; tr++) {
		memcpy(h_x, x1[tr], N*sizeof(float));
		cudaMemcpy(d_x, h_x, szxr, cudaMemcpyHostToDevice);
		GPU_AnalyticSignal ((cufftComplex *)d_xan1, d_x, N, &pin, &pout);
		
		memcpy(h_x, x2[tr], N*sizeof(float));
		cudaMemcpy(d_x, h_x, szxr, cudaMemcpyHostToDevice);
		GPU_AnalyticSignal ((cufftComplex *)d_xan2, d_x, N, &pin, &pout);
		
		// Invoke the kernel GPU_AmpNormf
		cuda_AmpNormf2(d_xan1, N, mem, NULL);
		cuda_AmpNormf2(d_xan2, N, mem, NULL);
		
		// Invoke the kernel PCC1_lowlevel
		blocksPerGrid = (L-l1 + threadsPerBlock - 1) / threadsPerBlock;
		PCC1_lowlevel2<<<blocksPerGrid, threadsPerBlock>>>(d_xan2, d_xan1, d_y, N, L, l1, Lag1);
		
		/* Copy result back */
		cudaMemcpy(h_y, d_y, szy, cudaMemcpyDeviceToHost);
		for (l=l1; l<L; l++) y[tr][l] = h_y[l];
	}
	
	/* Destroy the plans */
	cufftDestroy(pin);
	cufftDestroy(pout);
	
	cudaFree(d_xan1);
	cudaFree(d_xan2);
	cudaFree(d_y);
	cudaFree(d_x);
	
	cudaFreeHost(h_y);
	cudaFreeHost(h_x);
}
#else
void pcc1_highlevel2 (float **y, float **x1, float **x2, int N, unsigned int Tr, int Lag1, int Lag2) {
	cufftHandle pin, pout;
	float *d_x[2];
	float *h_x[2], *h_y, *d_y, *mem;
	float2 *d_xan[2];
	size_t szxr, szxc, szy;
	unsigned int tr, l, l1, L=(unsigned)abs(Lag2-Lag1+1);
	int threadsPerBlock = BLOCK_SIZE, blocksPerGrid1, blocksPerGrid2;
	
	if (Lag2 > N) L -= (Lag2-N);
	l1 = (Lag1 >= -N) ? 0 : -(Lag1+N);

	blocksPerGrid1 = (N + threadsPerBlock - 1) / threadsPerBlock;
	blocksPerGrid2 = (L-l1 + threadsPerBlock - 1) / threadsPerBlock;

	/* Convert data to float */
	szxr = N*sizeof(float);
	szxc = N*sizeof(float2);
	szy = L*sizeof(float);
	
	cudaMallocHost(&h_x[0], 2*szxr);
	h_x[1] = h_x[0] + N;
	cudaMallocHost(&h_y, szy);
	
	/* Allocate vectors in device memory */
	cudaMalloc(&d_x[0], 2*szxr);
	d_x[1] = d_x[0] + N;
	cudaMalloc(&d_xan[0], 2*szxc);
	d_xan[1] = d_xan[0] + N;
	cudaMalloc(&d_y, szy);
	cudaMalloc(&mem, blocksPerGrid1*sizeof(float));
	
	/* Make the plans */
	if (cufftPlan1d(&pin,  N, CUFFT_R2C, 2) != CUFFT_SUCCESS) { fprintf(stderr, "CUFFPlan1d failed\n"); return; }
	if (cufftPlan1d(&pout, N, CUFFT_C2C, 2) != CUFFT_SUCCESS) { fprintf(stderr, "CUFFPlan1d failed\n"); return; }
	
	for (tr=0; tr<Tr; tr++) {
		memcpy(h_x[0], x1[tr], N*sizeof(float));
		memcpy(h_x[1], x2[tr], N*sizeof(float));
		cudaMemcpy(d_x[0], h_x[0], 2*szxr, cudaMemcpyHostToDevice);
		cudaMemset(d_xan[0], 0, 2*szxc);
		
		/* out = FFT(in) */
		if (cufftExecR2C(pin, d_x[0], d_xan[0]) != CUFFT_SUCCESS) { fprintf(stderr, "cufftExecR2C failed\n"); return; }

		/* Make it analytic ( out(w<0) = 0, out(w>0) *= 2, out(0) & out(Nyquist) no change). */
		GPU_analytic<<<blocksPerGrid1, threadsPerBlock>>>(d_xan[0], N);
		GPU_analytic<<<blocksPerGrid1, threadsPerBlock>>>(d_xan[1], N);

		/* in = IFFT(out) */
		if (cufftExecC2C(pout, d_xan[0], d_xan[0], CUFFT_INVERSE) != CUFFT_SUCCESS) { fprintf(stderr, "cufftExecC2C failed\n"); return; }
		
		// Invoke the kernel GPU_AmpNormf
		cuda_AmpNormf2(d_xan[0], N, mem, NULL);
		cuda_AmpNormf2(d_xan[1], N, mem, NULL);
		
		// Invoke the kernel PCC1_lowlevel
		PCC1_lowlevel2<<<blocksPerGrid2, threadsPerBlock>>>(d_xan[1], d_xan[0], d_y, N, L, l1, Lag1);
		
		/* Copy result back */
		cudaMemcpy(h_y, d_y, szy, cudaMemcpyDeviceToHost);
		for (l=l1; l<L; l++) y[tr][l] = h_y[l];
	}
	
	/* Destroy the plans */
	cufftDestroy(pin);
	cufftDestroy(pout);
	
	cudaFree(d_xan[0]);
	cudaFree(d_y);
	cudaFree(d_x[0]);
	
	cudaFreeHost(h_y);
	cudaFreeHost(h_x[0]);
}
#endif

/**** Speedup WPCC2 */
__global__ void GPU_prod (float2 *y, float2 *x1, float2 *x2, float a, unsigned int N) {
	unsigned int n = blockDim.x * blockIdx.x + threadIdx.x;
	float2 ca1, ca2;
	
	if (n < N) {
		ca1 = x1[n];
		ca2 = x2[n];
		y[n].x = a*(ca1.x * ca2.x + ca1.y * ca2.y);
		y[n].y = a*(ca1.y * ca2.x - ca1.x * ca2.y);
	}
}

__global__ void GPU_cprod (float2 *y, float2 *x1, float2 *x2, float a, unsigned int N) {
	unsigned int n = blockDim.x * blockIdx.x + threadIdx.x;
	float2 ca1, ca2;
	
	if (n < N) {
		ca1 = x1[n];
		ca2 = x2[n];
		y[n].x += a*(ca1.x * ca2.x + ca1.y * ca2.y);
		y[n].y += a*(ca1.y * ca2.x - ca1.x * ca2.y);
	}
}

int cuda_tspcc2_set_freq (float **y0, float **x01, float **x02, unsigned int N, unsigned int Tr, unsigned int Nz, 
	const int Lag1, const int Lag2, unsigned int S, double *scale, _Complex double **wc, unsigned int *Ls, int *center, 
	unsigned int *Down_smp, float C) {
	
	cufftHandle plan;
	cudaError_t cudaerr;
	float2 *h_w, *w, *y, *x1, *x2, *x1_wt, *x2_wt, *pf2;
	float *mem, sfactor, fa2;
	float *x, *yr;
	_Complex double *w0, *pc;
	unsigned int s, Ls0, tr;
	int n, L, c, Lags, lag;
	int threadsPerBlock = BLOCK_SIZE, blocksPerGrid;
	size_t available=0, total=0;
	int num_gpus = 0, gpuid = -1;
	
	if (cudaSuccess != (cudaerr = cudaGetDeviceCount(&num_gpus) ))
		printf ("Error getting device count (%s)\n", cudaGetErrorString(cudaerr));

	for (n=0; n<num_gpus; n++) {
		if (num_gpus > 1) cudaSetDevice(n);
		if (cudaSuccess != (cudaerr = cudaGetDevice(&gpuid) ))
			printf ("Error getting device id (%s)\n", cudaGetErrorString(cudaerr));
		if (cudaSuccess != (cudaerr = cudaMemGetInfo(&available, &total) ))
			printf ("Error getting mem info (%s)\n", cudaGetErrorString(cudaerr));
		printf("num_gpus %d, CUDA device %d\n", num_gpus, gpuid);
		printf("Available memory = %ld bytes, Total memory = %ld bytes\n", available, total);
	}
	
	if (cudaerr != cudaSuccess) 
		{ printf ("Error getting memory info (%s)\n", cudaGetErrorString(cudaerr)); return -1; }
	
	if (Lag2 >= Lag1) {
		Lags = Lag2-Lag1+1;
		lag = Lag1;
	} else {
		Lags = Lag1-Lag2+1;
		lag = Lag2;
	}
	
	Ls0 = Ls[S-1];
	
	for (s=0; s<S; s++)
		if (Down_smp[s] != 1) return -2;
		
	cufftPlan1d(&plan, Nz, CUFFT_C2C, 1);
	
	/* FFTs of the wavelet family */
	cudaMallocHost(&h_w, Nz*sizeof(float2));
	cudaMalloc(&w, S*Nz*sizeof(float2));
	
	for (s=0; s<S; s++) {
		L = Ls[s];
		c =	center[s];
		w0 = wc[s];
		
		pc = w0+c;
		for (n=0; n<L-c; n++) {
			h_w[n].x = (float)creal(pc[n]);
			h_w[n].y = (float)cimag(pc[n]);
		}
		for (n=L-c; n<(signed)Nz-c; n++) {
			h_w[n].x = 0.;
			h_w[n].y = 0.;
		}
		pf2 = h_w + Nz - c;
		for (n=0; n<c; n++) {
			pf2[n].x = (float)creal(w0[n]);
			pf2[n].y = (float)cimag(w0[n]);
		}
		
		cudaMemcpy(w + s*Nz, h_w, Nz*sizeof(float2), cudaMemcpyHostToDevice);
		
		cufftExecC2C(plan, w + s*Nz, w + s*Nz, CUFFT_FORWARD);
	}
	
	cudaFreeHost(h_w);
	
	/* WPCC2 in freq. */
	blocksPerGrid = (Nz + threadsPerBlock - 1) / threadsPerBlock;
	
	cudaMalloc(&x  , N*sizeof(float));
	cudaMalloc(&x1, Nz*sizeof(float2));
	cudaMalloc(&x2, Nz*sizeof(float2));
	cudaMalloc(&x1_wt, Nz*sizeof(float2));
	cudaMalloc(&x2_wt, Nz*sizeof(float2));
	cudaMalloc(&y, Nz*sizeof(float2));
	cudaMalloc(&yr, Lags*sizeof(float));
	cudaMalloc(&mem, blocksPerGrid*sizeof(float));
	
	fa2 = 1./(float)Nz;
	for (tr=0; tr<Tr; tr++) {
		cudaMemcpy(x, x01[tr], N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemset(x1, 0, Nz*sizeof(float2));
		GPU_F2C<<<blocksPerGrid, threadsPerBlock>>>(x1 + Ls0, x, N);
		cufftExecC2C(plan, x1, x1, CUFFT_FORWARD);
		
		cudaMemcpy(x, x02[tr], N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemset(x2, 0, Nz*sizeof(float2));
		GPU_F2C<<<blocksPerGrid, threadsPerBlock>>>(x2 + Ls0, x, N);
		cufftExecC2C(plan, x2, x2, CUFFT_FORWARD);
		
		cudaMemset(y, 0, Nz*sizeof(float2));
		for (s=0; s<S; s++) {
			GPU_prod<<<blocksPerGrid, threadsPerBlock>>>(x1_wt, x1, w + s*Nz, fa2, Nz);
			cufftExecC2C(plan, x1_wt, x1_wt, CUFFT_INVERSE);
			cuda_AmpNormf3(x1_wt, Nz, mem, NULL);
			cufftExecC2C(plan, x1_wt, x1_wt, CUFFT_FORWARD);
			
			GPU_prod<<<blocksPerGrid, threadsPerBlock>>>(x2_wt, x2, w + s*Nz, fa2, Nz);
			cufftExecC2C(plan, x2_wt, x2_wt, CUFFT_INVERSE);
			cuda_AmpNormf3(x2_wt, Nz, mem, NULL);
			cufftExecC2C(plan, x2_wt, x2_wt, CUFFT_FORWARD);
			
			sfactor = 1 / (float)(scale[s] * (double)Nz); /* Regularization */
			GPU_cprod<<<blocksPerGrid, threadsPerBlock>>>(y, x2_wt, x1_wt, sfactor, Nz);
		}
		cufftExecC2C(plan, y, y, CUFFT_INVERSE);
		
		/* Copy the lag of interest and normalize */
		GPU_copy_and_normalized_lags<<<(Lags + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(yr, y, Nz, C, lag, Lags);
		cudaMemcpy(y0[tr], yr, Lags*sizeof(float), cudaMemcpyDeviceToHost);
	}
	/* Destroy plans */
	cufftDestroy(plan);
	
	/* Clean up */
	cudaFree(w);
	cudaFree(x);
	cudaFree(x1);
	cudaFree(x2);
	cudaFree(x1_wt);
	cudaFree(x2_wt);
	cudaFree(y);
	cudaFree(yr);
	cudaFree(mem);
	
	return 0;
}

