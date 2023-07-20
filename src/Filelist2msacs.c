/***********************************************************************/
/* Save a group of sac files into a single binary file to be use with  */
/* the PCC_fullpair code. This increases the average reading speed by  */
/* reducing latencies because of time acces in conventional hard disc. */
/*                                                                     */
/* Authors: Sergi Ventosa Rahuet (sergiventosa@hotmail.com)            */
/***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ReadManySacs.h"

int main_job (char *outfile, char *infiles, int Nmax);
void usage ();
int RDint    (int * const x, const char *str);

int main(int argc, char *argv[]) {
	char *filelist, *outfile;
	int Nmax = 0;
	
	if (argc < 3) usage();
	else {
		filelist = argv[1];
		outfile  = argv[2];
		if (argc > 3) {
			if (!strncmp(argv[3], "Nmax=", 5)) {
				if ( RDint(&Nmax, argv[3] + 5) ) {
					puts("Error when reading Nmax.\n"); Nmax = 0; 
				}
			}
		}
		main_job (outfile, filelist, Nmax);
	}
	
	return 0;
}

int main_job (char *outfile, char *filelist, int Nmax) {
	float **x=NULL;
	t_HeaderInfo *hdr=NULL;
	unsigned int Tr, N = Nmax;
	int nerr=0;
	float dt;
	
	nerr = ReadManySacs (&x, &hdr, NULL, &Tr, &N, &dt, filelist);
	if (nerr != 0) printf("Error %d when reading %s.\n", nerr, filelist);
	else {
		nerr = Write_ManySacsFile (x, hdr, Tr, N, outfile);
		if (nerr != 0) printf("Error %d when writing to %s.\n", nerr, outfile);
	}
	return nerr;
}

void usage () {
	puts("\nUSAGE: Filelist2msacs \"List of sac files\" \"Output file name\" [Nmax=\"maximum number of samples per sequence\"]");
}

int RDint (int * const x, const char *str) {
	char *pstr;
	
	*x = strtol(str, &pstr, 10);
	return (str == pstr) ? 1 : 0; 
}
