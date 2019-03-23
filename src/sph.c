/********************************************************************/
/* 5-6/4/2012: Basic operations in spherical coordinates.           */
/********************************************************************/
/* Sergi Ventosa Rahuet                                             */
/********************************************************************/

#include <math.h>
#include "sph.h"
#if 0 
/* Bug: gc[n] can be nan due to rounding errors that lead to abs(cos_gc) > 1. Fast solution. */
void sph_gcarc (double *gc, double latO, double lonO, double *latD, double *lonD, unsigned int N) {
	double cos_gc, sin_latO, cos_latO;
	unsigned int n;
	
	sin_latO = sin(latO);
	cos_latO = cos(latO);

	for (n=0; n<N; n++) {
		cos_gc = sin(latD[n]) * sin_latO + cos(latD[n]) * cos_latO * cos(lonO-lonD[n]);
		gc[n] = (cos_gc > 1.) ? 0. : acos(cos_gc);
	}
}

void sph_gcaz (double *gc, double *az, double latO, double lonO, double *latD, double *lonD, unsigned int N) {
	double cos_gc, sin_gc, sin_latO, cos_latO, sin_latD, cos_latD;
	unsigned int n;
	
	sin_latO = sin(latO);
	cos_latO = cos(latO);

	for (n=0; n<N; n++) {
		sin_latD = sin(latD[n]);
		cos_latD = cos(latD[n]);
		
		cos_gc = sin_latD * sin_latO + cos_latD * cos_latO * cos(lonO-lonD[n]);
		gc[n] = (cos_gc > 1.) ? 0. : acos(cos_gc);
		sin_gc = sin(gc[n]);
		
		az[n] = atan2(cos_latD * sin(lonD[n]-lonO) / sin_gc , (sin_latD - sin_latO * cos_gc)/(cos_latO * sin_gc) );
		if (az[n]<0) az[n] += 2*PI;
	}
}

#else /* Bug solution without using the ?: way. */
void sph_gcarc (double *gc, double latO, double lonO, double *latD, double *lonD, unsigned int N) {
	double cos_gc, da1;
	unsigned int n;

	for (n=0; n<N; n++) {
		da1 = sin((lonO - lonD[n])/2);
		da1 *= da1;
		cos_gc = cos(latD[n] - latO) * (1 - da1) - cos(latD[n] + latO) * da1;
		/* cos_gc = cos(latD[n] - latO) - 2*cos(latD[n]) * cos(latO) * da1; */ /* Idem as the line above. */
		gc[n] = acos(cos_gc);
	}
}

void sph_gcaz (double *gc, double *az, double latO, double lonO, double *latD, double *lonD, unsigned int N) {
	double da1, cos_gc, sin_gc, sin_latO, cos_latO, sin_latD, cos_latD;
	unsigned int n;
	
	sin_latO = sin(latO);
	cos_latO = cos(latO);
	
	#pragma omp parallel for default(shared) private(da1, sin_latD, cos_latD, cos_gc, sin_gc) schedule(static)
	for (n=0; n<N; n++) {
		sin_latD = sin(latD[n]);
		cos_latD = cos(latD[n]);
		
		da1 = sin((lonO - lonD[n])/2);
		da1 *= da1;
		cos_gc = cos(latD[n] - latO) * (1 - da1) - cos(latD[n] + latO) * da1;
		gc[n] = acos(cos_gc);
		sin_gc = sin(gc[n]);
		
		az[n] = atan2(cos_latD * sin(lonD[n]-lonO) / sin_gc , (sin_latD - sin_latO * cos_gc)/(cos_latO * sin_gc) );
		if (az[n]<0) az[n] += 2*PI;
	}
}

void sph_gcaz2 (double *gc, double *az, double *latO, double *lonO, double *latD, double *lonD, unsigned int N) {
	double da1, cos_gc, sin_gc, sin_latO, cos_latO, sin_latD, cos_latD;
	unsigned int n;
	
	#pragma omp parallel for default(shared) private(da1, sin_latO, cos_latO, sin_latD, cos_latD, cos_gc, sin_gc) schedule(static)
	for (n=0; n<N; n++) {
		sin_latO = sin(latO[n]);
		cos_latO = cos(latO[n]);
		sin_latD = sin(latD[n]);
		cos_latD = cos(latD[n]);
		
		da1 = sin((lonO[n] - lonD[n])/2);
		da1 *= da1;
		cos_gc = cos(latD[n] - latO[n]) * (1 - da1) - cos(latD[n] + latO[n]) * da1;
		gc[n] = acos(cos_gc);
		sin_gc = sin(gc[n]);
		
		az[n] = atan2(cos_latD * sin(lonD[n]-lonO[n]) / sin_gc , (sin_latD - sin_latO * cos_gc)/(cos_latO * sin_gc) );
		if (az[n]<0) az[n] += 2*PI;
	}
}

void sph_gcazbaz (double *gc, double *az, double *baz, double latO, double lonO, double *latD, double *lonD, unsigned int N) {
	double da1, cos_gc, sin_gc, sin_latO, cos_latO, sin_latD, cos_latD;
	unsigned int n;
	
	sin_latO = sin(latO);
	cos_latO = cos(latO);

	for (n=0; n<N; n++) {
		sin_latD = sin(latD[n]);
		cos_latD = cos(latD[n]);
		
		da1 = sin((lonO - lonD[n])/2);
		da1 *= da1;
		cos_gc = cos(latD[n] - latO) * (1 - da1) - cos(latD[n] + latO) * da1;
		/* cos_gc = cos(latD[n] - latO) - 2*cos(latD[n]) * cos(latO) * da1; */ /* Idem as the line above. */
		gc[n] = acos(cos_gc);
		sin_gc = sin(gc[n]);
		
		
		da1 = sin(lonD[n]-lonO);
		az[n]  = atan2 ( cos_latD * da1 / sin_gc , (sin_latD - sin_latO * cos_gc)/(cos_latO * sin_gc) );
		if (az[n]<0) az[n] += 2*PI;
		
		baz[n] = atan2 (-cos_latO * da1 / sin_gc , (sin_latO - sin_latD * cos_gc)/(cos_latD * sin_gc) );
		if (baz[n]<0) baz[n] += 2*PI;
	}
}

#endif

void sph_latlon (double *latD, double *lonD, double lat0, double lon0, double *gc, double *az, unsigned int N) {
	double E, F, H, aux, sin_gc, cos_gc, sin_lat, cos_lat, sin_az, cos_az;
	unsigned int n;
	
	sin_lat = sin(lat0);
	cos_lat = cos(lat0);
	for (n=0; n<N; n++) {
		sin_gc = sin(gc[n]);
		cos_gc = cos(gc[n]);
		sin_az = sin(az[n]);
		cos_az = cos(az[n]);
		
		E = sin_gc * sin_az;
		F = cos_gc * cos_lat - sin_gc * sin_lat * cos_az;
		H = cos_gc * sin_lat + sin_gc * cos_lat * cos_az;
		
		aux = lon0 + atan2(E, F);
		if (aux >  PI) aux -= 2*PI;
		if (aux < -PI) aux += 2*PI;
		lonD[n] = aux;
		
		latD[n] = PI/2 - atan2(sqrt(E*E + F*F), H);
	}
}
