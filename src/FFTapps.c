#include <complex.h>  /* When done before fftw3.h, makes fftw3.h use C99 complex types. */
#include <fftw3.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <semaphore.h>

//#define CUDAON
#ifdef CUDAON
#include "ccs_cuda.h"
#endif

#ifndef PI
#define PI 3.14159265358979323846
#endif

// #define FFTW_THREADS

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

int xcorr (double complex *y, double complex *x1, double complex *x2, unsigned int N) {
	fftw_plan pin1, pin2, pout;
	fftw_complex *out=NULL, *in1=NULL, *in2=NULL;
	double da1;
	unsigned int n, Nz=2*N-1, ua1, ua2;
	
	if (x1 == NULL || x2 == NULL) { y = NULL; return -1; }

	/* Memory allocation and zero padding. */
	Nz = 1 << (unsigned int)ceil(log2(Nz));
	if (NULL == (out = (fftw_complex *)fftw_malloc(Nz*sizeof(fftw_complex)) )) return -1;
	if (NULL == (in1 = (fftw_complex *)fftw_malloc(Nz*sizeof(fftw_complex)) )) return -1;
	if (NULL == (in2 = (fftw_complex *)fftw_malloc(Nz*sizeof(fftw_complex)) )) return -1;
	
	/* Make the plans */
	pin1 = fftw_plan_dft_1d(Nz, in1, in1, FFTW_FORWARD, FFTW_ESTIMATE);
	pin2 = fftw_plan_dft_1d(Nz, in2, in2, FFTW_FORWARD, FFTW_ESTIMATE);
	pout = fftw_plan_dft_1d(Nz, out, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	
	/* in1 = FFT(in1), in2 = FFT(in2) */
	for (n=0; n<N;  n++) in1[n] = x1[n];
	for (   ; n<Nz; n++) in1[n] = 0;
	for (n=0; n<N;  n++) in2[n] = x2[n];
	for (   ; n<Nz; n++) in2[n] = 0;
	
	fftw_execute(pin1);
	fftw_execute(pin2);

	/* out = IFFT(out) */
	for (n=0; n<Nz; n++) out[n] = in1[n]*conj(in2[n]);
	fftw_execute(pout);
	
	/* Destroy the plans */
	fftw_destroy_plan(pin1);
	fftw_destroy_plan(pin2);
	fftw_destroy_plan(pout);

	/* Position of the highest negative lag for inputs & output having an equal length. */
	da1 = 1/(double)Nz;
	ua1 = (N-1)/2;
	ua2 = Nz - ua1;
	for (n=0; n<ua1; n++) y[n] = da1 * out[ua2+n];
	for (    ; n<N;  n++) y[n] = da1 * out[n-ua1];
	
	fftw_free(in2);
	fftw_free(in1);
	fftw_free(out);
	
	return 0;
}

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

void AmpNormf (float complex *y, unsigned int N) {
	unsigned int n;
	float *pf1, fa1, fa2, fa3;

	/* for (n=0; n<N; n++) y[n] /= cabs(y[n]); */
	pf1 = (float *)y;
	for (n=0; n<2*N; n+=2) {
		fa1 = pf1[n];
		fa2 = pf1[n+1];
		fa3 = 1/sqrtf(fa1*fa1+fa2*fa2);
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

int pcc2_lowlevel (double * const y, double complex * const xan1, double complex * const xan2, const unsigned int N, const int Lag1, const int Lag2) {
	double complex *yfull=NULL;
	double da1;
	int er=0, L=Lag2-Lag1+1, lag;
	unsigned int l, ua1;
	
	if (NULL == (yfull = (double complex *)fftw_malloc(N*sizeof(double complex)) )) er = -2;
	
	if (!er) {
		xcorr (yfull, xan1, xan2, N);
		lag = Lag1 + (N-1)/2;
		if (lag < 0 || lag + L > N) return -3; /* Too large lags */
		ua1 = (unsigned)lag;
		da1 = 1/(double)N;
		for (l=0; l<(unsigned)L; l++) y[l] = da1 * creal(yfull[ua1 + l]);
	}
	
	fftw_free(yfull);
	
	return er;
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

void pcc1f_lowlevel (double * const y, float complex * const xan1, float complex * const xan2, const int N, const int Lag1, const int Lag2) {
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
		y[l] = da1/(2*(n2-n1));
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

void cc_lowlevel (double * const y, fftw_complex * const x1, fftw_complex * const x2, const int Nz, const int Lag1, const int Lag2, 
		fftw_plan *pout, double *out, fftw_complex *fout) {
	double da1 = 1/(double)Nz;
	unsigned int n, L, Nh = Nz/2 + 1;
	int lag;
	
	L = abs(Lag2-Lag1)+1;
	lag = (Lag2 >= Lag1) ? Lag1 : Lag2;
	
	for (n=0; n<Nh; n++) fout[n] = x1[n]*conj(x2[n]); /* the product        */
	fftw_execute(*pout);                               /* IFFT of the result */
	
	/* Copy the lag of interest and normalize */
	for (n=0; n<-lag; n++) y[n] = da1 * out[n+Nz+lag];
	for (   ; n<L;    n++) y[n] = da1 * out[n+lag];
}

void ccf_lowlevel (double * const y, fftwf_complex * const x1, fftwf_complex * const x2, const int Nz, const int Lag1, const int Lag2, 
		fftwf_plan *pout, float *out, fftwf_complex *fout) {
	double da1 = 1/(double)Nz;
	unsigned int n, L, Nh = Nz/2 + 1;
	int lag;
	
	L = abs(Lag2-Lag1)+1;
	lag = (Lag2 >= Lag1) ? Lag1 : Lag2;
	
	for (n=0; n<Nh; n++) fout[n] = x1[n]*conjf(x2[n]); /* the product        */
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

void gnf_lowlevel (double * const y, float * const x1, float * const x2, double norm1, double norm2, 
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

int pcc_set (double ** const y, double ** const x1, double ** const x2, const int N, const unsigned int Tr, const double v, const int Lag1, const int Lag2) {
	double complex *x1an, *x2an;
	unsigned int tr;
	int L=Lag2-Lag1+1, nerr = 0;
	
	if (Lag1 > N || Lag2 < -N) return 0;
	if (x1 == NULL || x2 == NULL || L < 0) return -1;
	
	/* Memory allocation */
	x1an = (double complex *)malloc(N*sizeof(double complex));
	x2an = (double complex *)malloc(N*sizeof(double complex));
	if (x1an == NULL || x2an == NULL) 
		nerr = -2; 
	else {
		for (tr=0; tr<Tr; tr++) {
			/* Phase signal (Analytic signal followed by amplitude normalization) of every x */
			AnalyticSignal (x1an, x1[tr], N);
			AmpNorm(x1an, N);
			AnalyticSignal (x2an, x2[tr], N);
			AmpNorm(x2an, N);
		
			/* Zero outputs. */
			memset(y[tr], 0, L*sizeof(double));
		
			/* The actual PCC computation */
			pcc_lowlevel (y[tr], x1an, x2an, N, v, Lag1, Lag2);
		}
	}
	/* Cleaning */
	free(x1an);
	free(x2an);
	
	return nerr;
}

int pcc1_set (double ** const y, double ** const x1, double ** const x2, const int N, const unsigned int Tr, const int Lag1, const int Lag2) {
	int L=Lag2-Lag1+1, nerr = 0;
	
	if (Lag1 > N || Lag2 < -N) return 0;
	if (x1 == NULL || x2 == NULL || L < 0) return -1;
	
	/* Memory allocation */
	#if 1
		sem_t *anok;
		unsigned int tr;
		float complex **fxa1, **fxa2;
		
		anok = (sem_t *)malloc(Tr*sizeof(sem_t));
		for (tr=0; tr<Tr; tr++) sem_init(&anok[tr], 0, 0);
		
		fxa1 = (float complex **)fftw_malloc(Tr*sizeof(float complex *));
		fxa1[0] = (float complex *)fftw_malloc(Tr*N*sizeof(float complex));
		for (tr=1; tr<Tr; tr++) fxa1[tr] = fxa1[tr-1] + N;
		
		fxa2 = (float complex **)fftw_malloc(Tr*sizeof(float complex *));
		fxa2[0] = (float complex *)fftw_malloc(Tr*N*sizeof(float complex));
		for (tr=1; tr<Tr; tr++) fxa2[tr] = fxa2[tr-1] + N;
		#ifdef CUDAON
			#pragma omp parallel sections
			{
				#pragma omp section
				{
					#pragma omp parallel 
					{
						fftwf_plan pain, paout;
						float *x;
						float complex *xa;
						unsigned int n, step;
						
						#pragma omp critical
						{
							x = (float *)fftw_malloc(N*sizeof(float));
							xa = (float complex *)fftw_malloc(N*sizeof(float complex));
							
							pain = fftwf_plan_dft_r2c_1d(N, x, xa, FFTW_ESTIMATE);
							paout = fftwf_plan_dft_1d(N, xa, xa, FFTW_BACKWARD, FFTW_ESTIMATE);
						}
						
						step = omp_get_num_threads();
						for (tr=omp_get_thread_num(); tr<Tr; tr+=step) {  /* First traces are processed first. */
							for (n=0; n<N; n++) x[n] = (float)x1[tr][n];
							AnalyticSignal_plan_float (xa, x, N, &pain, &paout);
							memcpy(fxa1[tr], xa, N*sizeof(float complex));
							
							for (n=0; n<N; n++) x[n] = (float)x2[tr][n];
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
					pcc1_highlevel (y, fxa1, fxa2, N, Tr, Lag1, Lag2, anok);
				}
			}
			for (tr=0; tr<Tr; tr++) sem_destroy(&anok[tr]);
			free(anok);
		#else
			#pragma omp parallel 
				{
					fftwf_plan pain, paout;
					float *x;
					float complex *xa;
					unsigned int n;
					
					#pragma omp critical
					{
						x = (float *)fftw_malloc(N*sizeof(float));
						xa = (float complex *)fftw_malloc(N*sizeof(float complex));
						
						pain = fftwf_plan_dft_r2c_1d(N, x, xa, FFTW_ESTIMATE);
						paout = fftwf_plan_dft_1d(N, xa, xa, FFTW_BACKWARD, FFTW_ESTIMATE);
					}
					
					#pragma omp for schedule(static)
					for (tr=0; tr<Tr; tr++) {
						for (n=0; n<N; n++) x[n] = (float)x1[tr][n];
						AnalyticSignal_plan_float (xa, x, N, &pain, &paout);
						memcpy(fxa1[tr], xa, N*sizeof(float complex));
						
						for (n=0; n<N; n++) x[n] = (float)x2[tr][n];
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
			#pragma omp parallel for schedule(static)
			for (tr=0; tr<Tr; tr++) {
				AmpNormf(fxa1[tr], N);
				AmpNormf(fxa2[tr], N);
				
				/* Zero outputs. */
				memset(y[tr], 0, L*sizeof(double));
			
				/* The actual PCC computation */
				pcc1f_lowlevel (y[tr], fxa1[tr], fxa2[tr], N, Lag1, Lag2);
			}
		#endif
		
		fftw_free(fxa1[0]);
		fftw_free(fxa1);
		fftw_free(fxa2[0]);
		fftw_free(fxa2);
	#else
		#ifdef CUDAON
			pcc1_highlevel2 (y, x1, x2, N, Tr, Lag1, Lag2);
		#else
			unsigned int tr;
			double complex *x1an = (double complex *)malloc(N*sizeof(double complex));
			double complex *x2an = (double complex *)malloc(N*sizeof(double complex));
			if (x1an == NULL || x2an == NULL) 
				nerr = -2; 
			else {
				for (tr=0; tr<Tr; tr++) {
					AnalyticSignal(x1an, x1[tr], N);
					AnalyticSignal(x2an, x2[tr], N);
					AmpNorm(x1an, N);
					AmpNorm(x2an, N);
					
					/* Zero outputs. */
					memset(y[tr], 0, L*sizeof(double));
				
					/* The actual PCC computation */
					pcc1_lowlevel (y[tr], x1an, x2an, N, Lag1, Lag2);
				}
			}
			/* Cleaning */
			free(x1an);
			free(x2an);
		#endif
	#endif
	
	return nerr;
}

int pcc2_set_float (double ** const y, double ** const x1, double ** const x2, const unsigned int N, const unsigned int Tr, const int Lag1, const int Lag2) {
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
		float *x, da1;
		fftwf_complex *xa1=NULL, *xa2=NULL, *out=NULL;
		unsigned int n, tr;
		
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
			da1 = 1/((double)Nz*(double)N); /* N*Nz may become a very high number */

			/* Analytic signal, zero padding and FFT of every x */
			#pragma omp for schedule(static)
			for (tr=0; tr<Tr; tr++) {
				/* Phase signal zero padding and the FFT before the actual xcorr */
				for (n=0; n<N; n++) x[n] = x1[tr][n];
				AnalyticSignal_plan_float (xa1, x, N, &pain1, &paout1);
				AmpNormf(xa1, N);
				for (n=N; n<Nz; n++) xa1[n] = 0;
				fftwf_execute(pin1);
				
				for (n=0; n<N; n++) x[n] = x2[tr][n];
				AnalyticSignal_plan_float (xa2, x, N, &pain2, &paout2);
				AmpNormf(xa2, N);
				for (n=N; n<Nz; n++) xa2[n] = 0;
				fftwf_execute(pin2);
				
				/* The actual xcorr */
				for (n=0; n<Nz; n++) out[n] = xa1[n]*conj(xa2[n]);  /* the product        */
				fftwf_execute(pout);                                /* IFFT of the result */
				
				/* Copy the lags of interest and normalize */
				for (n=0; n<-lag; n++) y[tr][n] = da1 * out[n+Nz+lag];
				for (   ; n<L;    n++) y[tr][n] = da1 * out[n+lag];
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
}

int pcc2_set (double ** const y, double ** const x1, double ** const x2, const unsigned int N, const unsigned int Tr, const int Lag1, const int Lag2) {
#if 1
	return pcc2_set_float (y, x1, x2, N, Tr, Lag1, Lag2);
#else
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
		fftw_plan pain1, paout1, pain2, paout2, pin1, pin2, pout;
		double *x, da1;
		fftw_complex *xa1=NULL, *xa2=NULL, *out=NULL;
		unsigned int n, tr;
		
		#pragma omp critical
		{
			x    = (double *)fftw_malloc(Nz*sizeof(double));
			xa1  = (fftw_complex *)fftw_malloc(Nz*sizeof(fftw_complex));
			xa2  = (fftw_complex *)fftw_malloc(Nz*sizeof(fftw_complex));
			out  = (fftw_complex *)fftw_malloc(Nz*sizeof(fftw_complex));
			
			pain1 = fftw_plan_dft_r2c_1d(N, x, xa1, FFTW_ESTIMATE);
			pain2 = fftw_plan_dft_r2c_1d(N, x, xa2, FFTW_ESTIMATE);
			paout1 = fftw_plan_dft_1d(N, xa1, xa1, FFTW_BACKWARD, FFTW_ESTIMATE);
			paout2 = fftw_plan_dft_1d(N, xa2, xa2, FFTW_BACKWARD, FFTW_ESTIMATE);
			pin1 = fftw_plan_dft_1d(Nz, xa1, xa1, FFTW_FORWARD, FFTW_ESTIMATE);
			pin2 = fftw_plan_dft_1d(Nz, xa2, xa2, FFTW_FORWARD, FFTW_ESTIMATE);
			pout = fftw_plan_dft_1d(Nz, out, out, FFTW_BACKWARD, FFTW_ESTIMATE); /* IFFT plan          */
		}
		
		if (xa1 != NULL && x2 != NULL && out != NULL) {
			da1 = 1/((double)Nz*(double)N); /* N*Nz may become a very high number */

			/* Analytic signal, zero padding and FFT of every x */
			#pragma omp for schedule(static)
			for (tr=0; tr<Tr; tr++) {
				/* Phase signal zero padding and the FFT before the actual xcorr */
				memcpy(x, x1[tr], N*sizeof(double));
				AnalyticSignal_plan (xa1, x, N, &pain1, &paout1);
				AmpNorm(xa1, N);
				for (n=N; n<Nz; n++) xa1[n] = 0;
				fftw_execute(pin1);
				
				memcpy(x, x2[tr], N*sizeof(double));
				AnalyticSignal_plan (xa2, x, N, &pain2, &paout2);
				AmpNorm(xa2, N);
				for (n=N; n<Nz; n++) xa2[n] = 0;
				fftw_execute(pin2);
				
				/* The actual xcorr */
				for (n=0; n<Nz; n++) out[n] = xa1[n]*conj(xa2[n]);  /* the product        */
				fftw_execute(pout);                                 /* IFFT of the result */
				
				/* Copy the lags of interest and normalize */
				for (n=0; n<-lag; n++) y[tr][n] = da1 * out[n+Nz+lag];
				for (   ; n<L;    n++) y[tr][n] = da1 * out[n+lag];
			}
		}
		
		#pragma omp critical
		{
			/* Destroy plans */
			fftw_destroy_plan(pout);
			fftw_destroy_plan(pin1);
			fftw_destroy_plan(pin2);
			fftw_destroy_plan(pain1);
			fftw_destroy_plan(pain2);
			fftw_destroy_plan(paout1);
			fftw_destroy_plan(paout2);
			
			/* Clean up */
			fftw_free(out);
			fftw_free(xa2);
			fftw_free(xa1);
			fftw_free(x);
		}
	}
	
	return nerr;
#endif
}

int ccgn_set (double ** const y, double ** const x1, double ** const x2, const unsigned int N, const unsigned int Tr, const int Lag1, const int Lag2) {
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
		double *in1, *in2, *out;
		double norm1, norm2;
		fftw_complex *fout, *fin1, *fin2;
		unsigned int n, tr;
		
		#pragma omp critical
		{
			in1 = (double *)fftw_malloc(Nz*sizeof(double));
			in2 = (double *)fftw_malloc(Nz*sizeof(double));
			out = (double *)fftw_malloc(Nz*sizeof(double));
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
				memcpy(in1, x1[tr], N*sizeof(double));
				for (n=N; n<Nz; n++) in1[n] = 0;    /* Zero padding */
				fftw_execute(pin1);                 /* FFT  */
				
				memcpy(in2, x2[tr], N*sizeof(double));
				for (n=N; n<Nz; n++) in2[n] = 0;    /* Zero padding */
				fftw_execute(pin2);                 /* FFT  */
				
				/* The actual xcorrs */
				cc_lowlevel (y[tr], fin1, fin2, Nz, Lag1, Lag2, &pout, out, fout);
				
				/* Normalized by || x1 || * || x2 ||  (on the overlapping part only) */
				norm1 = Norm (x1[tr], n11, n12);    /* norm of the first lag. */
				norm2 = Norm (x2[tr], n21, n22);    /*         "              */
				gn_lowlevel (y[tr], x1[tr], x2[tr], norm1, norm2, N, L, lag);
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
		}
	}
	
	return nerr;
}

int cc1b_set (double ** const y, double ** const x1, double ** const x2, const unsigned int N, const unsigned int Tr, const int Lag1, const int Lag2) {
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
		float *in1, *in2, *out;
		double norm1, norm2, *pd1;
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
				pd1 = x1[tr];
				for (n=0; n<N; n++)  in1[n] = (pd1[n] >= 0) ? 1 : -1;
				for (n=N; n<Nz; n++) in1[n] = 0;    /* Zero padding */
				fftwf_execute(pin1);                /* FFT  */
				
				pd1 = x2[tr];
				for (n=0; n<N; n++)  in2[n] = (pd1[n] >  0) ? 1 : -1;
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
