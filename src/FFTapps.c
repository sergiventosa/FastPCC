#include <complex.h>  /* When done before fftw3.h, makes fftw3.h use C99 complex types. */
#include <fftw3.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <semaphore.h>
#include "wavelet_v7.h"
#include "cdotx.h"


//#define CUDAON
#ifdef CUDAON
#include "ccs_cuda.h"
#endif

#ifndef PI
#define PI 3.14159265358979323846
#endif

// #define FFTW_THREADS
void F2D_vec (double *y, float *x, unsigned int N) {
	unsigned int n;
	
	for (n=0; n<N; n++) y[n] = (double)x[n];
}

void D2F_vec (float *y, double *x, unsigned int N) {
	unsigned int n;
	
	for (n=0; n<N; n++) y[n] = (float)x[n];
}

double std_vec (double complex *x, unsigned int N) {
	double out=0;
	unsigned int n;
	
	if (x==NULL) { printf("std_vec: NULL input pointer\n"); return 0; }
	
	for (n=0; n<N; n++) out += x[n]*conj(x[n]);
	return sqrt(out)/N;
}

int AnalyticSignal (double complex *y, double *x, unsigned int N) {
	fftw_plan pin, pout;
	double da1;
	unsigned int n;

	if (x == NULL) { y = NULL; return -1; }

	/* Make the plans */
	#pragma omp critical
	{
		pin = fftw_plan_dft_r2c_1d(N, x, y, FFTW_ESTIMATE);
		pout = fftw_plan_dft_1d(N, y, y, FFTW_BACKWARD, FFTW_ESTIMATE);
	}
	
	/* out = FFT(in) */
	fftw_execute(pin);

	/* Make it analytic ( out(w<0) = 0, out(w>0) *= 2, out(0) & out(Nyquist) no change). */
	da1 = 1/(double)N; /* For the normalization. */
	y[0] *= da1;
	for (n=1; n<(N+1)>>1; n++) y[n] *= 2*da1;
	if (!(N&1)) y[n++] *= da1;
	for (; n<N; n++) y[n] = 0; /* Because of fftw_plan_r2c_1d() doesn't define them. */ 

	/* in = IFFT(out) */
	fftw_execute(pout);
	
	/* Destroy the plans */
	#pragma omp critical
	{
		fftw_destroy_plan(pin);
		fftw_destroy_plan(pout);
	}
	
	return 0;
}

int AnalyticSignal_plan (double complex *y, double *x, unsigned int N, fftw_plan *pin, fftw_plan *pout) {
	double da1;
	unsigned int n;

	if (x == NULL) { y = NULL; return -1; }

	/* out = FFT(in) */
	fftw_execute(*pin);

	/* Make it analytic ( out(w<0) = 0, out(w>0) *= 2, out(0) & out(Nyquist) no change). */
	da1 = 1/(double)N; /* For the normalization. */
	y[0] *= da1;
	for (n=1; n<(N+1)>>1; n++) y[n] *= 2*da1;
	if (!(N&1)) y[n++] *= da1;
	for (; n<N; n++) y[n] = 0; /* Because of fftw_plan_r2c_1d() doesn't define them. */ 

	/* in = IFFT(out) */
	fftw_execute(*pout);
	
	return 0;
}

int AnalyticSignal_plan_float (float complex *y, float *x, unsigned int N, fftwf_plan *pin, fftwf_plan *pout) {
	float da1;
	unsigned int n;

	if (x == NULL) { y = NULL; return -1; }

	/* out = FFT(in) */
	fftwf_execute(*pin);

	/* Make it analytic ( out(w<0) = 0, out(w>0) *= 2, out(0) & out(Nyquist) no change). */
	da1 = 1/(double)N; /* For the normalization. */
	y[0] *= da1;
	for (n=1; n<(N+1)>>1; n++) y[n] *= 2*da1;
	if (!(N&1)) y[n++] *= da1;
	for (; n<N; n++) y[n] = 0; /* Because of fftw_plan_r2c_1d() doesn't define them. */ 

	/* in = IFFT(out) */
	fftwf_execute(*pout);
	
	return 0;
}

#if 0 /* Numerically unstable when data is not exactly zero due to, e.g., finite precision errors. */
void AmpNorm (double complex *y, unsigned int N) {
	unsigned int n;
	double *pd1, da1, da2, da3;

	/* for (n=0; n<N; n++) y[n] /= cabs(y[n]); */
	pd1 = (double *)y;
	for (n=0; n<2*N; n+=2) {
		da1 = pd1[n];
		da2 = pd1[n+1];
		da3 = 1/sqrt(da1*da1+da2*da2);
		if (isnormal(da3)) {
			pd1[n]   *= da3;
			pd1[n+1] *= da3;
		} else {
			pd1[n]   = 0;
			pd1[n+1] = 0;
		}
	}
}
#else
void AmpNorm (double complex *y, unsigned int N) {
	unsigned int n;
	double *pd1, eps, da1, da2, da3;

	/* for (n=0; n<N; n++) y[n] /= cabs(y[n]); */
	pd1 = (double *)y;
	eps = 0;
	for (n=0; n<2*N; n+=2) {
		da1 = pd1[n];
		da2 = pd1[n+1];
		da3 = da1*da1+da2*da2;
		if (eps < da3) eps = da3;
	}
	eps *= 1e-6;
	// printf("%9.4g\n", eps);
	
	for (n=0; n<2*N; n+=2) {
		da1 = pd1[n];
		da2 = pd1[n+1];
		da3 = 1/sqrt(da1*da1+da2*da2 + eps);
		if (isnormal(da3)) {
			pd1[n]   *= da3;
			pd1[n+1] *= da3;
		} else {
			pd1[n]   = 0;
			pd1[n+1] = 0;
		}
	}
}
#endif

void AmpNormf (float complex *y, unsigned int N) {
	unsigned int n;
	float *pf1, eps, fa1, fa2, fa3;

	/* for (n=0; n<N; n++) y[n] /= cabs(y[n]); */
	pf1 = (float *)y;
	eps = 0;
	for (n=0; n<2*N; n+=2) {
		fa1 = pf1[n];
		fa2 = pf1[n+1];
		fa3 = fa1*fa1+fa2*fa2;
		if (eps < fa3) eps = fa3;
	}
	eps = 1e-6*sqrt(eps);
	
	for (n=0; n<2*N; n+=2) {
		fa1 = pf1[n];
		fa2 = pf1[n+1];
		fa3 = 1/(sqrtf(fa1*fa1+fa2*fa2) + eps);
		if (isnormal(fa3)) {
			pf1[n]   *= fa3;
			pf1[n+1] *= fa3;
		} else {
			pf1[n]   = 0;
			pf1[n+1] = 0;
		}
	}
}

double Norm (double * const x, const unsigned int n1, const unsigned int n2) {
	double norm = 0;
	unsigned int n;

	for (n=n1; n<n2; n++) norm += x[n]*x[n];
	return norm;
}

double Normf (float * const x, const unsigned int n1, const unsigned int n2) {
	double norm = 0;
	unsigned int n;

	for (n=n1; n<n2; n++) norm += x[n]*x[n];
	return norm;
}

void pcc1_lowlevel (double * const y, double complex * const xan1, double complex * const xan2, const int N, const int Lag1, const int Lag2) {
	double da1, da2[2], da3[2], *pd1, *pd2;
	int L=Lag2-Lag1+1, lag;
	unsigned int n, n1, n2, l, l1;
	
	if (Lag2 > N) L -= (Lag2-N);
	l1 = (Lag1 >= -N) ? 0 : -(Lag1+N);
	
	#pragma omp parallel for default(shared) private(lag, n, n1, n2, da1, da2, da3, pd1, pd2) schedule(dynamic,10)
	for (l=l1; l<L; l++) {
		lag = Lag1 + l;
		if (lag < 0) { n1 = -lag; n2 = N;     }
		else         { n1 = 0;    n2 = N-lag; }
		
		pd1 = (double *)(xan1 + n1 + lag);
		pd2 = (double *)(xan2 + n1);
		da1 = 0;
		for (n=0; n<2*(n2-n1); n+=2) {
			da2[0] = pd1[n]   + pd2[n];
			da2[1] = pd1[n+1] + pd2[n+1];
			da3[0] = pd1[n]   - pd2[n];
			da3[1] = pd1[n+1] - pd2[n+1];
			da1 += sqrtf(da2[0]*da2[0]+da2[1]*da2[1]) - sqrtf(da3[0]*da3[0]+da3[1]*da3[1]); /* Using sqrtf to speed up things a bit */
		}
		y[l] = da1/(2*(n2-n1));
	}
}

void pcc1f_lowlevel (float * const y, float complex * const xan1, float complex * const xan2, const int N, const int Lag1, const int Lag2) {
	double da1;
	float fa2[2], fa3[2], *pf1, *pf2;
	int L=Lag2-Lag1+1, lag;
	unsigned int n, n1, n2, l, l1;
	
	if (Lag2 > N) L -= (Lag2-N);
	l1 = (Lag1 >= -N) ? 0 : -(Lag1+N);
	
	// #pragma omp parallel for default(shared) private(lag, n, n1, n2, da1, fa2, fa3, pf1, pf2) schedule(dynamic,10)
	for (l=l1; l<L; l++) {
		lag = Lag1 + l;
		if (lag < 0) { n1 = -lag; n2 = N;     }
		else         { n1 = 0;    n2 = N-lag; }
		
		pf1 = (float *)(xan1 + n1 + lag);
		pf2 = (float *)(xan2 + n1);
		da1 = 0;
		for (n=0; n<2*(n2-n1); n+=2) {
			fa2[0] = pf1[n]   + pf2[n];
			fa2[1] = pf1[n+1] + pf2[n+1];
			fa3[0] = pf1[n]   - pf2[n];
			fa3[1] = pf1[n+1] - pf2[n+1];
			da1 += sqrtf(fa2[0]*fa2[0]+fa2[1]*fa2[1]) - sqrtf(fa3[0]*fa3[0]+fa3[1]*fa3[1]);
		}
		y[l] = (float)da1/(2*(n2-n1));
	}
}

void pcc_lowlevel (double * const y, double complex * const xan1, double complex * const xan2, const int N, const double v, const int Lag1, const int Lag2) {
	double da1, da2[2], da3[2], *pd1, *pd2;
	double V=v/2;
	int L=Lag2-Lag1+1, lag;
	unsigned int n, n1, n2, l, l1;
	
	if (Lag2 > N) L -= (Lag2-N);
	l1 = (Lag1 >= -N) ? 0 : -(Lag1+N);
	
	#pragma omp parallel for default(shared) private(lag, n, n1, n2, da1, da2, da3, pd1, pd2) schedule(dynamic,10)
	for (l=l1; l<L; l++) {
		lag = Lag1 + l;
		if (lag < 0) { n1 = -lag; n2 = N;     }
		else         { n1 = 0;    n2 = N-lag; }
		
		pd1 = (double *)(xan1 + n1 + lag);
		pd2 = (double *)(xan2 + n1);
		da1 = 0;
		for (n=0; n<2*(n2-n1); n+=2) {
			da2[0] = pd1[n]   + pd2[n];
			da2[1] = pd1[n+1] + pd2[n+1];
			da3[0] = pd1[n]   - pd2[n];
			da3[1] = pd1[n+1] - pd2[n+1];
			da1 += powf(da2[0]*da2[0]+da2[1]*da2[1], V) - powf(da3[0]*da3[0]+da3[1]*da3[1], V); /* Using powf to speed up things a bit */
		}
		y[l] = da1/(pow(2,V)*(n2-n1));
	}
}

void pccf_lowlevel (float * const y, float complex * const xan1, float complex * const xan2, const int N, const double v, const int Lag1, const int Lag2) {
	float V=v/2, fa2[2], fa3[2], *pf1, *pf2;
	double da1;
	int L=Lag2-Lag1+1, lag;
	unsigned int n, n1, n2, l, l1;
	
	if (Lag2 > N) L -= (Lag2-N);
	l1 = (Lag1 >= -N) ? 0 : -(Lag1+N);
	
	#pragma omp parallel for default(shared) private(lag, n, n1, n2, da1, fa2, fa3, pf1, pf2) schedule(dynamic,10)
	for (l=l1; l<L; l++) {
		lag = Lag1 + l;
		if (lag < 0) { n1 = -lag; n2 = N;     }
		else         { n1 = 0;    n2 = N-lag; }
		
		pf1 = (float *)(xan1 + n1 + lag);
		pf2 = (float *)(xan2 + n1);
		da1 = 0;
		for (n=0; n<2*(n2-n1); n+=2) {
			fa2[0] = pf1[n]   + pf2[n];
			fa2[1] = pf1[n+1] + pf2[n+1];
			fa3[0] = pf1[n]   - pf2[n];
			fa3[1] = pf1[n+1] - pf2[n+1];
			da1 += powf(fa2[0]*fa2[0]+fa2[1]*fa2[1], V) - powf(fa3[0]*fa3[0]+fa3[1]*fa3[1], V); /* Using powf to speed up things a bit */
		}
		y[l] = da1/(pow(2,V)*(n2-n1));
	}
}

void cc_lowlevel (double * const y, fftw_complex * const x1, fftw_complex * const x2, const int Nz, const int Lag1, const int Lag2, 
		fftw_plan *pout, double *out, fftw_complex *fout) {
	double da1 = 1./(double)Nz;
	unsigned int L, Nh = Nz/2 + 1;
	int n, lag;
	
	L = abs(Lag2-Lag1)+1;
	lag = (Lag2 >= Lag1) ? Lag1 : Lag2;
	
	for (n=0; n<Nh; n++) fout[n] = conj(x1[n])*x2[n]; /* the product        */
	fftw_execute(*pout);                               /* IFFT of the result */
	
	/* Copy the lag of interest and normalize */
	for (n=0; n<-lag; n++) y[n] = da1 * out[n+Nz+lag];
	for (   ; n<L;    n++) y[n] = da1 * out[n+lag];
}

void ccf_lowlevel (float * const y, fftwf_complex * const x1, fftwf_complex * const x2, const int Nz, const int Lag1, const int Lag2, 
		fftwf_plan *pout, float *out, fftwf_complex *fout) {
	double da1 = 1./(double)Nz;
	unsigned int L, Nh = Nz/2 + 1;
	int n, lag;
	
	L = abs(Lag2-Lag1)+1;
	lag = (Lag2 >= Lag1) ? Lag1 : Lag2;
	
	for (n=0; n<Nh; n++) fout[n] = conjf(x1[n])*x2[n]; /* the product        */
	fftwf_execute(*pout);                               /* IFFT of the result */
	
	/* Copy the lag of interest and normalize */
	for (n=0; n<-lag; n++) y[n] = da1 * out[n+Nz+lag];
	for (   ; n<L;    n++) y[n] = da1 * out[n+lag];
}

void gn_lowlevel (double * const y, double * const x1, double * const x2, double norm1, double norm2, 
		unsigned int N, unsigned int L, int lag) {
	unsigned int n, n11=0, n21=0, n12=N, n22=N;
	
	if (lag > 0) { 
		n21  = (unsigned)lag;
		n12 -= (unsigned)lag;
	} else {
		n11  = (unsigned)(-lag);
		n22 -= (unsigned)(-lag);
	}
	
	for (n=0; n<L-1 && n22<N; n++) {
		y[n] /= sqrt(norm1 * norm2);
		n11--;
		norm1 += x1[n11]*x1[n11];
		norm2 += x2[n22]*x2[n22];
		n22++;
	}
	for (  ; n<L-1; n++) {
		y[n] /= sqrt(norm1 * norm2);
		n12--;
		norm1 -= x1[n12]*x1[n12];
		norm2 -= x2[n21]*x2[n21];
		n21++;
	}
	y[n] /= sqrt(norm1 * norm2);
}

void gnf_lowlevel (float * const y, float * const x1, float * const x2, double norm1, double norm2, 
		unsigned int N, unsigned int L, int lag) {
	unsigned int n, n11=0, n21=0, n12=N, n22=N;
	
	if (lag > 0) { 
		n21  = (unsigned)lag;
		n12 -= (unsigned)lag;
	} else {
		n11  = (unsigned)(-lag);
		n22 -= (unsigned)(-lag);
	}
	
	for (n=0; n<L-1 && n22<N; n++) {
		y[n] /= sqrt(norm1 * norm2);
		n11--;
		norm1 += x1[n11]*x1[n11];
		norm2 += x2[n22]*x2[n22];
		n22++;
	}
	for (  ; n<L-1; n++) {
		y[n] /= sqrt(norm1 * norm2);
		n12--;
		norm1 -= x1[n12]*x1[n12];
		norm2 -= x2[n21]*x2[n21];
		n21++;
	}
	y[n] /= sqrt(norm1 * norm2);
	
}

#if 0
int pcc_set (float ** const y, float ** const x1, float ** const x2, const int N, const unsigned int Tr, const double v, const int Lag1, const int Lag2) {
	double complex *x1an, *x2an;
	double *xd, *yd;
	unsigned int tr;
	int L=Lag2-Lag1+1, nerr = 0;
	
	if (Lag1 > N || Lag2 < -N) return 0;
	if (x1 == NULL || x2 == NULL || L < 0) return -1;
	
	/* Memory allocation */
	xd = (double *)malloc(N*sizeof(double));
	yd = (double *)malloc(N*sizeof(double));
	x1an = (double complex *)malloc(N*sizeof(double complex));
	x2an = (double complex *)malloc(N*sizeof(double complex));
	if (x1an == NULL || x2an == NULL || xd == NULL || yd == NULL) 
		nerr = -2; 
	else {
		for (tr=0; tr<Tr; tr++) {
			/* Phase signal (Analytic signal followed by amplitude normalization) of every x */
			F2D_vec(xd, x1[tr], N);
			AnalyticSignal (x1an, xd, N);
			AmpNorm(x1an, N);
			
			F2D_vec(xd, x2[tr], N);
			AnalyticSignal (x2an, xd, N);
			AmpNorm(x2an, N);
		
			/* Zero outputs. */
			memset(y[tr], 0, L*sizeof(double));
		
			/* The actual PCC computation */
			pcc_lowlevel (yd, x2an, x1an, N, v, Lag1, Lag2);
			D2F_vec(y[tr], yd, L);
		}
	}
	/* Cleaning */
	free(x2an);
	free(x1an);
	free(yd);
	free(xd);
	return nerr;
}
#else
int pcc_set (float ** const y, float ** const x1, float ** const x2, const int N, const unsigned int Tr, const double v, const int Lag1, const int Lag2) {
	int L=Lag2-Lag1+1, nerr = 0, OnTheCPU;
	
	if (Lag1 > N || Lag2 < -N) return 0;
	if (x1 == NULL || x2 == NULL || L < 0) return -1;
	
	/* Memory allocation */
	unsigned int tr;
	float complex **fxa1, **fxa2;
	
	fxa1 = (float complex **)fftw_malloc(Tr*sizeof(float complex *));
	fxa1[0] = (float complex *)fftw_malloc(Tr*N*sizeof(float complex));
	for (tr=1; tr<Tr; tr++) fxa1[tr] = fxa1[tr-1] + N;
	
	fxa2 = (float complex **)fftw_malloc(Tr*sizeof(float complex *));
	fxa2[0] = (float complex *)fftw_malloc(Tr*N*sizeof(float complex));
	for (tr=1; tr<Tr; tr++) fxa2[tr] = fxa2[tr-1] + N;
	#ifdef CUDAON
		sem_t *anok;
		
		anok = (sem_t *)malloc(Tr*sizeof(sem_t));
		for (tr=0; tr<Tr; tr++) sem_init(&anok[tr], 0, 0);
		
		omp_set_nested(1);
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				#pragma omp parallel 
				{
					fftwf_plan pain, paout;
					float *x;
					float complex *xa;
					
					#pragma omp critical
					{
						x = (float *)fftw_malloc(N*sizeof(float));
						xa = (float complex *)fftw_malloc(N*sizeof(float complex));
						
						pain = fftwf_plan_dft_r2c_1d(N, x, xa, FFTW_ESTIMATE);
						paout = fftwf_plan_dft_1d(N, xa, xa, FFTW_BACKWARD, FFTW_ESTIMATE);
					}
					
					#pragma omp for schedule(static,16)
					for (tr=0; tr<Tr; tr++) {
						memcpy(x, x1[tr], N*sizeof(float));
						AnalyticSignal_plan_float (xa, x, N, &pain, &paout);
						memcpy(fxa1[tr], xa, N*sizeof(float complex));
						
						memcpy(x, x2[tr], N*sizeof(float));
						AnalyticSignal_plan_float (xa, x, N, &pain, &paout);
						memcpy(fxa2[tr], xa, N*sizeof(float complex));
						
						sem_post(&anok[tr]);
					}
					
					#pragma omp critical
					{
						fftw_free(x);
						fftw_free(xa);
						
						fftwf_destroy_plan(pain);
						fftwf_destroy_plan(paout);
					}
				}
			}
			#pragma omp section
			{
				OnTheCPU = ( 0 == pcc_highlevel (y, fxa1, fxa2, N, Tr, v, Lag1, Lag2, anok) ) ? 0 : 1;
			}
		}
		for (tr=0; tr<Tr; tr++) sem_destroy(&anok[tr]);
		free(anok);
	#else
		OnTheCPU = 1;
		
		#pragma omp parallel 
		{
			fftwf_plan pain, paout;
			float *x;
			float complex *xa;
			
			#pragma omp critical
			{
				x = (float *)fftw_malloc(N*sizeof(float));
				xa = (float complex *)fftw_malloc(N*sizeof(float complex));
				
				pain = fftwf_plan_dft_r2c_1d(N, x, xa, FFTW_ESTIMATE);
				paout = fftwf_plan_dft_1d(N, xa, xa, FFTW_BACKWARD, FFTW_ESTIMATE);
			}
			
			#pragma omp for schedule(static)
			for (tr=0; tr<Tr; tr++) {
				memcpy(x, x1[tr], N*sizeof(float));
				AnalyticSignal_plan_float (xa, x, N, &pain, &paout);
				memcpy(fxa1[tr], xa, N*sizeof(float complex));
				
				memcpy(x, x2[tr], N*sizeof(float));
				AnalyticSignal_plan_float (xa, x, N, &pain, &paout);
				memcpy(fxa2[tr], xa, N*sizeof(float complex));
			}
			
			#pragma omp critical
			{
				fftw_free(x);
				fftw_free(xa);
				
				fftwf_destroy_plan(pain);
				fftwf_destroy_plan(paout);
			}
		}
	#endif
	
	if (OnTheCPU) {
		#pragma omp parallel for schedule(static)
		for (tr=0; tr<Tr; tr++) {
			AmpNormf(fxa1[tr], N);
			AmpNormf(fxa2[tr], N);
			
			/* Zero outputs. */
			memset(y[tr], 0, L*sizeof(float));
		
			/* The actual PCC computation */
			pccf_lowlevel (y[tr], fxa1[tr], fxa2[tr], N, v, Lag1, Lag2);
		}
	}
	
	fftw_free(fxa1[0]);
	fftw_free(fxa1);
	fftw_free(fxa2[0]);
	fftw_free(fxa2);
	
	return nerr;
}
#endif

int pcc1_set (float ** const y, float ** const x1, float ** const x2, const int N, const unsigned int Tr, const int Lag1, const int Lag2) {
	int L=Lag2-Lag1+1, nerr = 0, OnTheCPU;
	
	if (Lag1 > N || Lag2 < -N) return 0;
	if (x1 == NULL || x2 == NULL || L < 0) return -1;
	
	/* Memory allocation */
	#if 1
		unsigned int tr;
		float complex **fxa1, **fxa2;
		
		fxa1 = (float complex **)fftw_malloc(Tr*sizeof(float complex *));
		fxa1[0] = (float complex *)fftw_malloc(Tr*N*sizeof(float complex));
		for (tr=1; tr<Tr; tr++) fxa1[tr] = fxa1[tr-1] + N;
		
		fxa2 = (float complex **)fftw_malloc(Tr*sizeof(float complex *));
		fxa2[0] = (float complex *)fftw_malloc(Tr*N*sizeof(float complex));
		for (tr=1; tr<Tr; tr++) fxa2[tr] = fxa2[tr-1] + N;
		#ifdef CUDAON
			sem_t *anok;
			
			anok = (sem_t *)malloc(Tr*sizeof(sem_t));
			for (tr=0; tr<Tr; tr++) sem_init(&anok[tr], 0, 0);
			
			omp_set_nested(1);
			#pragma omp parallel sections
			{
				#pragma omp section
				{
					#pragma omp parallel 
					{
						fftwf_plan pain, paout;
						float *x;
						float complex *xa;
						
						#pragma omp critical
						{
							x = (float *)fftw_malloc(N*sizeof(float));
							xa = (float complex *)fftw_malloc(N*sizeof(float complex));
							
							pain = fftwf_plan_dft_r2c_1d(N, x, xa, FFTW_ESTIMATE);
							paout = fftwf_plan_dft_1d(N, xa, xa, FFTW_BACKWARD, FFTW_ESTIMATE);
						}
						
						#pragma omp for schedule(static,16)
						for (tr=0; tr<Tr; tr++) {
							memcpy(x, x1[tr], N*sizeof(float));
							AnalyticSignal_plan_float (xa, x, N, &pain, &paout);
							memcpy(fxa1[tr], xa, N*sizeof(float complex));
							
							memcpy(x, x2[tr], N*sizeof(float));
							AnalyticSignal_plan_float (xa, x, N, &pain, &paout);
							memcpy(fxa2[tr], xa, N*sizeof(float complex));
							
							sem_post(&anok[tr]);
						}
						
						#pragma omp critical
						{
							fftw_free(x);
							fftw_free(xa);
							
							fftwf_destroy_plan(pain);
							fftwf_destroy_plan(paout);
						}
					}
				}
				#pragma omp section
				{
					OnTheCPU = ( 0 == pcc1_highlevel (y, fxa1, fxa2, N, Tr, Lag1, Lag2, anok) ) ? 0 : 1;
				}
			}
			for (tr=0; tr<Tr; tr++) sem_destroy(&anok[tr]);
			free(anok);
		#else
			OnTheCPU = 1;
			
			#pragma omp parallel 
			{
				fftwf_plan pain, paout;
				float *x;
				float complex *xa;
				
				#pragma omp critical
				{
					x = (float *)fftw_malloc(N*sizeof(float));
					xa = (float complex *)fftw_malloc(N*sizeof(float complex));
					
					pain = fftwf_plan_dft_r2c_1d(N, x, xa, FFTW_ESTIMATE);
					paout = fftwf_plan_dft_1d(N, xa, xa, FFTW_BACKWARD, FFTW_ESTIMATE);
				}
				
				#pragma omp for schedule(static)
				for (tr=0; tr<Tr; tr++) {
					memcpy(x, x1[tr], N*sizeof(float));
					AnalyticSignal_plan_float (xa, x, N, &pain, &paout);
					memcpy(fxa1[tr], xa, N*sizeof(float complex));
					
					memcpy(x, x2[tr], N*sizeof(float));
					AnalyticSignal_plan_float (xa, x, N, &pain, &paout);
					memcpy(fxa2[tr], xa, N*sizeof(float complex));
				}
				
				#pragma omp critical
				{
					fftw_free(x);
					fftw_free(xa);
					
					fftwf_destroy_plan(pain);
					fftwf_destroy_plan(paout);
				}
			}
		#endif
		
		if (OnTheCPU) {
			#pragma omp parallel for schedule(static)
			for (tr=0; tr<Tr; tr++) {
				AmpNormf(fxa1[tr], N);
				AmpNormf(fxa2[tr], N);
				
				/* Zero outputs. */
				memset(y[tr], 0, L*sizeof(float));
			
				/* The actual PCC computation */
				pcc1f_lowlevel (y[tr], fxa2[tr], fxa1[tr], N, Lag1, Lag2);
			}
		}
		
		fftw_free(fxa1[0]);
		fftw_free(fxa1);
		fftw_free(fxa2[0]);
		fftw_free(fxa2);
	#else
		#ifdef CUDAON
			pcc1_highlevel2 (y, x1, x2, N, Tr, Lag1, Lag2);
		#else
			unsigned int tr;
			double xd = (double *)malloc(N*sizeof(double));
			double yd = (double *)malloc(N*sizeof(double));
			double complex *x1an = (double complex *)malloc(N*sizeof(double complex));
			double complex *x2an = (double complex *)malloc(N*sizeof(double complex));
			
			if (x1an == NULL || x2an == NULL) 
				nerr = -2; 
			else {
				for (tr=0; tr<Tr; tr++) {
					F2D_vec(xd, x1[tr], N);
					AnalyticSignal(x1an, xd, N);
					AmpNorm(x1an, N);
					
					F2D_vec(xd, x1[tr], N);
					AnalyticSignal(x2an, xd, N);
					AmpNorm(x2an, N);
					
					/* Zero outputs. */
					memset(y[tr], 0, L*sizeof(double));
				
					/* The actual PCC computation */
					pcc1_lowlevel (yd, x2an, x1an, N, Lag1, Lag2);
					D2F_vec(y[tr], yd, L);
					
				}
			}
			/* Cleaning */
			free(x2an);
			free(x1an);
			free(yd);
			free(xd);
		#endif
	#endif
	
	return nerr;
}

int pcc2_set (float ** const y, float ** const x1, float ** const x2, const unsigned int N, const unsigned int Tr, const int Lag1, const int Lag2) {
//#ifdef CUDAON
//	return cuda_pcc2_set (y, x1, x2, N, Tr, Lag1, Lag2);
//#else
	unsigned int Nz, M, ua1, ua2;
	int L, lag, nerr=0;
	
	if (Lag2 >= Lag1) {
		L=Lag2-Lag1+1;
		lag = Lag1;
	} else {
		L=Lag1-Lag2+1;
		lag = Lag2;
	}

	/* Memory allocation */
	ua1 = abs(Lag1);
	ua2 = abs(Lag2);
	if (ua1 >= N || ua2 >= N) return -3; /* Too large lags */
	M = (ua1 > ua2) ? ua1 : ua2;
	Nz = 1 << (unsigned int)ceil(log2(N+M)); /* Because the lags higher than M are rejected */
	
	#pragma omp parallel
	{
		fftwf_plan pain1, paout1, pain2, paout2, pin1, pin2, pout;
		float *x, fa1;
		fftwf_complex *xa1=NULL, *xa2=NULL, *out=NULL;
		unsigned int tr;
		int n;
		
		#pragma omp critical
		{
			x    = (float *)fftw_malloc(Nz*sizeof(float));
			xa1  = (fftwf_complex *)fftw_malloc(Nz*sizeof(fftwf_complex));
			xa2  = (fftwf_complex *)fftw_malloc(Nz*sizeof(fftwf_complex));
			out  = (fftwf_complex *)fftw_malloc(Nz*sizeof(fftwf_complex));
			
			pain1 = fftwf_plan_dft_r2c_1d(N, x, xa1, FFTW_ESTIMATE);
			pain2 = fftwf_plan_dft_r2c_1d(N, x, xa2, FFTW_ESTIMATE);
			paout1 = fftwf_plan_dft_1d(N, xa1, xa1, FFTW_BACKWARD, FFTW_ESTIMATE);
			paout2 = fftwf_plan_dft_1d(N, xa2, xa2, FFTW_BACKWARD, FFTW_ESTIMATE);
			pin1 = fftwf_plan_dft_1d(Nz, xa1, xa1, FFTW_FORWARD, FFTW_ESTIMATE);
			pin2 = fftwf_plan_dft_1d(Nz, xa2, xa2, FFTW_FORWARD, FFTW_ESTIMATE);
			pout = fftwf_plan_dft_1d(Nz, out, out, FFTW_BACKWARD, FFTW_ESTIMATE); /* IFFT plan          */
		}
		
		if (xa1 != NULL && x2 != NULL && out != NULL) {
			fa1 = 1/((float)Nz*(float)N); /* N*Nz may become a very high number */

			/* Analytic signal, zero padding and FFT of every x */
			#pragma omp for schedule(static)
			for (tr=0; tr<Tr; tr++) {
				/* Phase signal zero padding and the FFT before the actual xcorr */
				memcpy(x, x1[tr], N*sizeof(float));
				AnalyticSignal_plan_float (xa1, x, N, &pain1, &paout1);
				AmpNormf(xa1, N);
				for (n=N; n<Nz; n++) xa1[n] = 0;
				fftwf_execute(pin1);
				
				memcpy(x, x2[tr], N*sizeof(float));
				AnalyticSignal_plan_float (xa2, x, N, &pain2, &paout2);
				AmpNormf(xa2, N);
				for (n=N; n<Nz; n++) xa2[n] = 0;
				fftwf_execute(pin2);
				
				/* The actual xcorr */
				for (n=0; n<Nz; n++) out[n] = conj(xa1[n])*xa2[n];  /* the product        */
				fftwf_execute(pout);                                /* IFFT of the result */
				
				/* Copy the lags of interest and normalize */
				for (n=0; n<-lag; n++) y[tr][n] = fa1 * out[n+Nz+lag];
				for (   ; n<L;    n++) y[tr][n] = fa1 * out[n+lag];
			}
		}
		
		#pragma omp critical
		{
			/* Destroy plans */
			fftwf_destroy_plan(pout);
			fftwf_destroy_plan(pin1);
			fftwf_destroy_plan(pin2);
			fftwf_destroy_plan(pain1);
			fftwf_destroy_plan(pain2);
			fftwf_destroy_plan(paout1);
			fftwf_destroy_plan(paout2);
			
			/* Clean up */
			fftw_free(out);
			fftw_free(xa2);
			fftw_free(xa1);
			fftw_free(x);
		}
	}
	
	return nerr;
//#endif
}

int ccgn_set (float ** const y, float ** const x1, float ** const x2, const unsigned int N, const unsigned int Tr, const int Lag1, const int Lag2) {
	unsigned int Nz, M, Nh, ua1, ua2;
	unsigned int n11=0, n21=0, n12=N, n22=N;
	int L, lag, nerr=0;
	
	if (Lag2 >= Lag1) {
		L=Lag2-Lag1+1;
		lag = Lag1;
	} else {
		L=Lag1-Lag2+1;
		lag = Lag2;
	}
	
	if (lag > 0) { 
		n21  = (unsigned)lag;
		n12 -= (unsigned)lag;
	} else {
		n11  = (unsigned)(-lag);
		n22 -= (unsigned)(-lag);
	}
	
	/* Memory allocation */
	ua1 = abs(Lag1);
	ua2 = abs(Lag2);
	if (ua1 >= N || ua2 >= N) return -3; /* Too large lags */
	M = (ua1 > ua2) ? ua1 : ua2; 
	Nz = 1 << (unsigned int)ceil(log2(N+M));  /* Because the lags higher than M are rejected */
	Nh = Nz/2 + 1;  /* Number of complex used in r2c & c2r ffts. */
	
	#pragma omp parallel
	{
		fftw_plan pin1, pin2, pout;
		double *in1, *in2, *out, *yd;
		double norm1, norm2;
		fftw_complex *fout, *fin1, *fin2;
		unsigned int n, tr;
		
		#pragma omp critical
		{
			in1 = (double *)fftw_malloc(Nz*sizeof(double));
			in2 = (double *)fftw_malloc(Nz*sizeof(double));
			out = (double *)fftw_malloc(Nz*sizeof(double));
			yd  = (double *)fftw_malloc(L*sizeof(double));
			fin1 = (fftw_complex *)fftw_malloc(Nh*sizeof(fftw_complex));
			fin2 = (fftw_complex *)fftw_malloc(Nh*sizeof(fftw_complex));
			fout = (fftw_complex *)fftw_malloc(Nh*sizeof(fftw_complex));
			
			/* Make plans */
			pin1 = fftw_plan_dft_r2c_1d(Nz, in1, fin1, FFTW_ESTIMATE); /* FFT plan  */
			pin2 = fftw_plan_dft_r2c_1d(Nz, in2, fin2, FFTW_ESTIMATE); /*    "      */
			pout = fftw_plan_dft_c2r_1d(Nz, fout, out, FFTW_ESTIMATE); /* IFFT plan */
		}
		
		if (in1 != NULL && in2 != NULL && out != NULL && fin1 != NULL && fin2 != NULL && fout != NULL) {
			
			#pragma omp for schedule(static)
			for (tr=0; tr<Tr; tr++) {
				F2D_vec(in1, x1[tr], N);
				for (n=N; n<Nz; n++) in1[n] = 0;    /* Zero padding */
				fftw_execute(pin1);                 /* FFT  */
				
				F2D_vec(in2, x2[tr], N);
				for (n=N; n<Nz; n++) in2[n] = 0;    /* Zero padding */
				fftw_execute(pin2);                 /* FFT  */
				
				/* The actual xcorrs */
				cc_lowlevel (yd, fin1, fin2, Nz, Lag1, Lag2, &pout, out, fout);
				
				/* Normalized by || x1 || * || x2 ||  (on the overlapping part only) */
				norm1 = Norm (in1, n11, n12);    /* norm of the first lag. */
				norm2 = Norm (in2, n21, n22);    /*         "              */
				gn_lowlevel (yd, in1, in2, norm1, norm2, N, L, lag);
				D2F_vec(y[tr], yd, L);
			}
		}
		
		#pragma omp critical
		{
			/* Destroy the FFT & IFFT plans */
			fftw_destroy_plan(pout);
			fftw_destroy_plan(pin1);
			fftw_destroy_plan(pin2);
	
			/* Clean up */
			fftw_free(fout);
			fftw_free(fin2);
			fftw_free(fin1);
			fftw_free(in1);
			fftw_free(in2);
			fftw_free(out);
			fftw_free(yd);
		}
	}
	
	return nerr;
}

int cc1b_set (float ** const y, float ** const x1, float ** const x2, const unsigned int N, const unsigned int Tr, const int Lag1, const int Lag2) {
	unsigned int Nz, M, Nh, ua1, ua2;
	unsigned int n11=0, n21=0, n12=N, n22=N;
	int L, lag, nerr=0;
	
	if (Lag2 >= Lag1) {
		L=Lag2-Lag1+1;
		lag = Lag1;
	} else {
		L=Lag1-Lag2+1;
		lag = Lag2;
	}
	
	if (lag > 0) { 
		n21  = (unsigned)lag;
		n12 -= (unsigned)lag;
	} else {
		n11  = (unsigned)(-lag);
		n22 -= (unsigned)(-lag);
	}
	
	/* Memory allocation */
	/* Nz=2*N-1; */
	/* Nz = 1 << (unsigned int)ceil(log2(Nz)); */
	ua1 = abs(Lag1);
	ua2 = abs(Lag2);
	if (ua1 >= N || ua2 >= N) return -3; /* Too large lags */
	M = (ua1 > ua2) ? ua1 : ua2; 
	Nz = 1 << (unsigned int)ceil(log2(N+M));  /* Because the lags higher than M are rejected */
	Nh = Nz/2 + 1;  /* Number of complex used in r2c & c2r ffts. */
	
	#pragma omp parallel
	{
		fftwf_plan pin1, pin2, pout;
		float *in1, *in2, *out, *pf1;
		double norm1, norm2;
		fftwf_complex *fout, *fin1, *fin2;
		unsigned n, tr;
		
		#pragma omp critical
		{
			in1 = (float *)fftw_malloc(Nz*sizeof(float));
			in2 = (float *)fftw_malloc(Nz*sizeof(float));
			out = (float *)fftw_malloc(Nz*sizeof(float));
			fin1 = (fftwf_complex *)fftw_malloc(Nh*sizeof(fftwf_complex));
			fin2 = (fftwf_complex *)fftw_malloc(Nh*sizeof(fftwf_complex));
			fout = (fftwf_complex *)fftw_malloc(Nh*sizeof(fftwf_complex));
			
			pin1 = fftwf_plan_dft_r2c_1d(Nz, in1, fin1, FFTW_ESTIMATE); /* FFT plan  */
			pin2 = fftwf_plan_dft_r2c_1d(Nz, in2, fin2, FFTW_ESTIMATE); /*    "      */
			pout = fftwf_plan_dft_c2r_1d(Nz, fout, out, FFTW_ESTIMATE); /* IFFT plan */
		}
	
		if (in1 != NULL && in2 != NULL && out != NULL && fin1 != NULL && fin2 != NULL && fout != NULL) {
			#pragma omp for schedule(static)
			for (tr=0; tr<Tr; tr++) {
				pf1 = x1[tr];
				for (n=0; n<N; n++)  in1[n] = (pf1[n] >= 0) ? 1 : -1;
				for (n=N; n<Nz; n++) in1[n] = 0;    /* Zero padding */
				fftwf_execute(pin1);                /* FFT  */
				
				pf1 = x2[tr];
				for (n=0; n<N; n++)  in2[n] = (pf1[n] >  0) ? 1 : -1;
				for (n=N; n<Nz; n++) in2[n] = 0;    /* Zero padding */
				fftwf_execute(pin2);                /* FFT  */
				
				/* The actual xcorrs */
				ccf_lowlevel (y[tr], fin1, fin2, Nz, Lag1, Lag2, &pout, out, fout);
				
				/* Normalized by || x1 || * || x2 ||  (on the overlapping part only) */
				norm1 = Normf (in1, n11, n12);    /* norm of the first lag. */
				norm2 = Normf (in2, n21, n22);    /*         "              */
				gnf_lowlevel (y[tr], in1, in2, norm1, norm2, N, L, lag);
			}
		}
		
		#pragma omp critical
		{
			 /* Destroy the FFT & IFFT plans */
			fftwf_destroy_plan(pout);
			fftwf_destroy_plan(pin1);
			fftwf_destroy_plan(pin2);
			
			/* Clean up */
			fftw_free(fout);
			fftw_free(fin2);
			fftw_free(fin1);
			fftw_free(in1);
			fftw_free(in2);
			fftw_free(out);
		}
	}
	return nerr;
}

/* Frequency domain version. */
int tspcc2_set (float ** const y, float ** const x1, float ** const x2, const unsigned int N, 
		const unsigned int Tr, const int Lag1, const int Lag2, double pmin, double pmax, unsigned int V, int type, double op1) {
	unsigned int J, S;
	unsigned int Nz, M, m, ua1, ua2, Ls0;
	double s0, b0, K0;
	int L, nerr = 0;
	t_WaveletFamily *pWF;
	
	L = abs(Lag2-Lag1+1);
	
	/* Memory allocation */
	ua1 = abs(Lag1);
	ua2 = abs(Lag2);
	if (ua1 >= N || ua2 >= N) return -3; /* Too large lags */
	M = (ua1 > ua2) ? ua1 : ua2;
	Nz = 1 << (unsigned int)ceil(log2(N+M)); /* Because the lags higher than M are rejected */
	
	/* Zero outputs. */
	for (int tr=0; tr<Tr; tr++) memset(y[tr], 0, L*sizeof(float));
	
	/* Wavelet initializations */
	/* Downsampling is not supported yet. So, b0 is not used. */ 
	if (type == -3) { /*    MexHat    */
		s0 = pmin/(sqrt(2)*PI);
		b0 = 0.5;
	} else if (type == -1 || type == -2) { /*    Morlet    */
		if (op1==0) op1 = PI*sqrt(2/log(2));  /* w0 parameter */
		// op1 = PI*sqrt(2/log(2));  /* w0 parameter */
		s0 = pmin*op1/(2*PI);
		b0 = 0.5; // b0 = (unsigned int)pow( 2, round(log2(s0)) );
	} else return -3;
	J = (unsigned int)round(1./(double)V + log(pmax/pmin)/log(2));
	pWF = CreateWaveletFamily (type, J, V, Nz, s0, b0, 0, op1, 1);
	
	S = V*J;
	Ls0 = pWF->Ls[S-1];
	if (Nz-(N+M) < Ls0) Nz *= 2; /* The longest wavelet (Ls0) is limited to Nz, so Nz *= 2 is enough. */
	
	K0 = 0;
	for (m=0; m<S; m++) K0 += 1/pWF->scale[m];
	
	#ifdef CUDAON
	{
		/* float C = ( log(pWF->a0) / (2 * pWF->Cpsi * pWF->V) ) / (double)N; */
		float C = 1./(K0*(double)N);
		
		cuda_tspcc2_set_freq(y, x1, x2, N, Tr, Nz, Lag1, Lag2, S, pWF->scale, pWF->wframe.wc, pWF->Ls, pWF->center, pWF->Down_smp, C);
	}
	#else
	{
		fftw_plan pw;
		double complex *pc, **fw;
		int c, Ls, lag;
		
		lag = (Lag2 >= Lag1) ? Lag1 : Lag2;
		
		fw = (fftw_complex **)malloc(S*sizeof(fftw_complex *));
		fw[0] = (fftw_complex *)fftw_malloc(S*Nz*sizeof(fftw_complex));
		for (int s=1; s<S; s++) fw[s] = fw[s-1] + Nz;
		
		/* FFTs of the Wavelet Family */
		for (int s=0; s<S; s++) {
			pw = fftw_plan_dft_1d(Nz, fw[s], fw[s], FFTW_FORWARD, FFTW_ESTIMATE);
			
			pc = pWF->wframe.wc[s];
			c  = pWF->center[s];
			Ls = pWF->Ls[s]; 
			memcpy(fw[s],        pc+c, (Ls-c)*sizeof(fftw_complex));
			memset(fw[s] + Ls-c, 0,    (Nz-Ls)*sizeof(fftw_complex));
			memcpy(fw[s] + Nz-c, pc,   c*sizeof(fftw_complex));
			
			fftw_execute(pw);
			fftw_destroy_plan(pw);
		}
		
		#pragma omp parallel
		{
			fftw_plan pin1, pin2, py_wt, px1_wt, px2_wt, px1_iwt, px2_iwt;
			double complex *in1, *in2, *x1_wt, *x2_wt, *y_wt, *pc1;
			double da2, da3, C;
			float *pf1;
			int tr, s, n;
			
			#pragma omp critical
			{
				/* Initializations */
				in1 = (fftw_complex *)fftw_malloc(Nz*sizeof(fftw_complex));
				in2 = (fftw_complex *)fftw_malloc(Nz*sizeof(fftw_complex));
				x1_wt = (fftw_complex *)fftw_malloc(Nz*sizeof(fftw_complex));
				x2_wt = (fftw_complex *)fftw_malloc(Nz*sizeof(fftw_complex));
				y_wt  = (fftw_complex *)fftw_malloc(Nz*sizeof(fftw_complex));
				
				/* Create plans */
				pin1 = fftw_plan_dft_1d(Nz, in1, in1, FFTW_FORWARD, FFTW_ESTIMATE);
				pin2 = fftw_plan_dft_1d(Nz, in2, in2, FFTW_FORWARD, FFTW_ESTIMATE);
				px1_iwt = fftw_plan_dft_1d(Nz, x1_wt, x1_wt, FFTW_BACKWARD, FFTW_ESTIMATE);
				px2_iwt = fftw_plan_dft_1d(Nz, x2_wt, x2_wt, FFTW_BACKWARD, FFTW_ESTIMATE);
				px1_wt  = fftw_plan_dft_1d(Nz, x1_wt, x1_wt, FFTW_FORWARD,  FFTW_ESTIMATE);
				px2_wt  = fftw_plan_dft_1d(Nz, x2_wt, x2_wt, FFTW_FORWARD,  FFTW_ESTIMATE);
				py_wt   = fftw_plan_dft_1d(Nz,  y_wt,  y_wt, FFTW_BACKWARD, FFTW_ESTIMATE); /* IFFT plan          */
			}
			
			/* C = ( log(pWF->a0) / (2 * pWF->Cpsi * pWF->V) )  / (double)N; */
			C = 1./(K0*(double)N);
			da2 = 1./(double)Nz;
			
			#pragma omp for schedule(static)
			for (tr=0; tr<Tr; tr++) {
				/* Decomposition */
				pf1 = x1[tr];
				memset(in1, 0, Ls0*sizeof(fftw_complex));
				pc1 = in1 + Ls0;
				for (n=0; n<N; n++) pc1[n] = da2*pf1[n];
				memset(pc1 + N, 0, (Nz-N-Ls0)*sizeof(fftw_complex));
				fftw_execute(pin1);
				
				pf1 = x2[tr];
				memset(in2, 0, Ls0*sizeof(fftw_complex));
				pc1 = in2 + Ls0;
				for (n=0; n<N; n++) pc1[n] = da2*pf1[n];
				memset(pc1 + N, 0, (Nz-N-Ls0)*sizeof(fftw_complex));
				fftw_execute(pin2);
				
				/* BPFs + PCCs + Lazy Inverse */
				memset(y_wt, 0, Nz*sizeof(fftw_complex));
				for (s=0; s<S; s++) {
					pc1 = fw[s];
					/* CWT of in1 at the scale s */
					for (n=0; n<Nz; n++) x1_wt[n] = in1[n]*pc1[n];
					fftw_execute(px1_iwt);
					AmpNorm(x1_wt, Nz);
					fftw_execute(px1_wt);
					
					/* CWT of in2 at the scale s */
					for (n=0; n<Nz; n++) x2_wt[n] = in2[n]*pc1[n];
					fftw_execute(px2_iwt);
					AmpNorm(x2_wt, Nz);
					fftw_execute(px2_wt);
					
					/* Product & Lazy inverse */
					// for (n=0; n<Nz; n++) y_wt[n] += x1_wt[n]*conj(x2_wt[n]); /* the product        */
					da3 = 1./(pWF->scale[s] * (double)Nz);
					for (n=0; n<Nz; n++) y_wt[n] += da3 * conj(x1_wt[n])*x2_wt[n];  /* the product      */
				} 
				fftw_execute(py_wt);                                         /* IFFT of the result */
				
				/* Copy the lag of interest and normalize */
				for (n=0; n<-lag; n++) y[tr][n] = C * creal(y_wt[n+Nz+lag]);
				for (   ; n<L;    n++) y[tr][n] = C * creal(y_wt[n+lag]);
			}
			
			#pragma omp critical
			{
				/* Destroy plans */
				fftw_destroy_plan(pin1);
				fftw_destroy_plan(pin2);
				fftw_destroy_plan(px1_wt);
				fftw_destroy_plan(px2_wt);
				fftw_destroy_plan(px1_iwt);
				fftw_destroy_plan(px2_iwt);
				fftw_destroy_plan(py_wt);
			
				/* Cleaning */
				fftw_free(x1_wt);
				fftw_free(x2_wt);
				fftw_free(y_wt);
				fftw_free(in1);
				fftw_free(in2);
			}
		}
		
		fftw_free(fw[0]);
		free(fw);
	}
	#endif
	
	DestroyWaveletFamily (pWF);

	return nerr;
}

