#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ReadManySacs.h"

int main_job (char *outfile, char *infiles);
void usage ();

int main(int argc, char *argv[]) {
	char *filelist, *outfile;
	
	if (argc < 2) usage();
	else {
		filelist = argv[1];
		outfile  = argv[2];
		main_job (outfile, filelist);
	}
	
	return 0;
}

int main_job (char *outfile, char *filelist) {
	double **x=NULL;
	t_HeaderInfo *hdr=NULL;
	unsigned int Tr, N;
	int nerr=0;
	float dt;
	
	nerr = ReadManySacs (&x, &hdr, &Tr, &N, &dt, filelist);
	// printf("sta=%s\n", hdr->sta);
	if (nerr != 0) printf("Error %d when reading %s.", nerr, filelist);
	else {
		nerr = Write_ManySacsFile (x, hdr, Tr, N, outfile);
		if (nerr != 0) printf("Error %d when writing to %s.", nerr, outfile);
	}
	return nerr;
}

void usage () {
	puts("\nUSAGE: Filelist2msacs \"List of sac files\" \"Output file name\"");
}
