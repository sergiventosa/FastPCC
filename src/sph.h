#ifndef SPH_H
#define SPH_H

#ifndef PI
#define PI 3.14159265358979328
#endif
#ifndef DEG2RAD
#define DEG2RAD PI/180.
#endif
#ifndef RAD2DEG
#define RAD2DEG 180./PI
#endif

/* Parameters definition: */
/* gc is distance.        */
/* az is azimuth defined at (lat0,lon0), usually the epicenter location.    (Angle from the north pole towards the station direction).   */
/* baz is backazimuth defined at (latD,lonD), usually the station location. (Angle from the north pole towards the epicenter direction). */
/* lat0 and lon0 are the latitud and longitud of the origin, usually the epicenter location.      */
/* latD and lonD are the latitud and longitud of the "destination", usually the station location. */
void sph_gcarc (double *gc, double latO, double lonO, double *latD, double *lonD, unsigned int N);
void sph_gcaz (double *gc, double *az, double latO, double lonO, double *latD, double *lonD, unsigned int N);
void sph_gcaz2 (double *gc, double *az, double *latO, double *lonO, double *latD, double *lonD, unsigned int N);
void sph_gcazbaz (double *gc, double *az, double *baz, double latO, double lonO, double *latD, double *lonD, unsigned int N);
void sph_latlon (double *latD, double *lonD, double lat0, double lon0, double *gc, double *az, unsigned int N);

#endif
