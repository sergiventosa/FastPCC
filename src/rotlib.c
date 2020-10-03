/*****************************************************************************/
/* Library for a few rotation operations.                                    */
/*                                                                           */
/* Authors: Sergi Ventosa Rahuet (sergiventosa@hotmail.com)                  */
/*                                                                           */
/*****************************************************************************/
#include "rotlib.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "sph.h"

void CorrectRevesedPolarity (float **x, unsigned int N, unsigned int Tr, t_HeaderInfo *Header) {
	unsigned int n, tr; 
	float cmpinc, cmpaz, *px;
	
	for (tr=0; tr<Tr; tr++) {
		cmpinc = Header[tr].cmpinc;
		cmpaz  = Header[tr].cmpaz;
		px = x[tr];
		
		if (cmpinc == 180) {
			for (n=0; n<N; n++) px[n] = -px[n];
			Header[tr].cmpinc = 0;
		} else if (cmpinc == 90) {
			if (cmpaz < 315 && cmpaz > 135) {
				for (n=0; n<N; n++) px[n] = -px[n];
				Header[tr].cmpaz -= 180;
				if (Header[tr].cmpaz < 0) Header[tr].cmpaz += 360;
			}
		}
	}
}
