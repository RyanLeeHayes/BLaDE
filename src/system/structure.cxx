#include <string.h>

#include "system/system.h"
#include "system/structure.h"
#include "main/defines.h"
#include "io/io.h"

void parse_structure(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  FILE *fp;

  // All routines except reset need a strcture, so just initialize it and save time.
  if (system->structure==NULL) {
    system->structure=new Structure();
  }

  io_nexta(line,token);
  if (strcmp(token,"reset")==0) {
    if (system->structure) {
      delete(system->structure);
      system->structure=NULL;
    }
  } else if (strcmp(token,"file")==0) {
    io_nexta(line,token);
    if (strcmp(token,"psf")==0) {
      io_nexta(line,token);
      fp=fpopen(token,"r");
      system->structure->add_structure_psf_file(fp);
      fclose(fp);
    } else {
      fatal(__FILE__,__LINE__,"Unsupported structure file format: %s\n",token);
    }
  } else if (strcmp(token,"print")==0) {
    system->structure->dump();
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized token: %s\n",token); // FIXIT add token name
  }
}

void Structure::add_structure_psf_file(FILE *fp)
{
  char line[MAXLENGTHSTRING];
  char token[MAXLENGTHSTRING];
  int i,j;

  // "read" header"
  fgets(line, MAXLENGTHSTRING, fp);
  io_nexta(line,token);
  if (strcmp(token,"PSF")!=0) {
    fatal(__FILE__,__LINE__,"First line of PSF must start with PSF\n");
  }

  // "Read" title
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf number of title lines");
  for (i=0; i<j; i++) {
    fgets(line, MAXLENGTHSTRING, fp);
  }

  // Read atoms
  fgets(line, MAXLENGTHSTRING, fp);
  atomCount=io_nexti(line,fp,"psf number of atoms");
#warning "Memory leak if already initialized"
  atomIdx=(int*)calloc(atomCount,sizeof(int));
  segName=new std::string[atomCount];
  resIdx=(int*)calloc(atomCount,sizeof(int));
  resName=new std::string[atomCount];
  atomName=new std::string[atomCount];
  atomTypeName=new std::string[atomCount];
  atomTypeIdx=(int*)calloc(atomCount,sizeof(int));
  charge=(double*)calloc(atomCount,sizeof(double));
  mass=(double*)calloc(atomCount,sizeof(double));
  for (i=0; i<atomCount; i++) {
    fgets(line, MAXLENGTHSTRING, fp);
    j=io_nexti(line)-1;
    if (i!=j) {
      fatal(__FILE__,__LINE__,"Found atom %d when atom %d expected\n",j,i);
    }
    atomIdx[i]=i;
    segName[i]=io_nexts(line);
    resIdx[i]=io_nexti(line);
    resName[i]=io_nexts(line);
    atomName[i]=io_nexts(line);
    io_nexts(line);
    // atomTypeName[i]=io_nexts(line);
    charge[i]=io_nextf(line);
    mass[i]=io_nextf(line);
  }

  // Read bonds
  fgets(line, MAXLENGTHSTRING, fp);
  bondCount=io_nexti(line,fp,"psf number of bonds");
#warning "Memory leak if already initialized"
  bonds=(int(*)[2])calloc(bondCount,sizeof(int[2]));
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<bondCount; i++) {
    for (j=0; j<2; j++) {
      bonds[i][j]=io_nexti(line,fp,"psf bond atom")-1;
      if (bonds[i][j]>=atomCount || bonds[i][j]<0) {
        fatal(__FILE__,__LINE__,"Atom %d in bond %d is out of range\n",bonds[i][j],i);
      }
    }
  }
  
  // Read angles
  fgets(line, MAXLENGTHSTRING, fp);
  angleCount=io_nexti(line,fp,"psf number of angles");
#warning "Memory leak if already initialized"
  angles=(int(*)[3])calloc(angleCount,sizeof(int[3]));
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<angleCount; i++) {
    for (j=0; j<3; j++) {
      angles[i][j]=io_nexti(line,fp,"psf angle atom")-1;
      if (angles[i][j]>=atomCount || angles[i][j]<0) {
        fatal(__FILE__,__LINE__,"Atom %d in angle %d is out of range\n",angles[i][j],i);
      }
    }
  }
  
  // Read dihes
  fgets(line, MAXLENGTHSTRING, fp);
  diheCount=io_nexti(line,fp,"psf number of dihedrals");
#warning "Memory leak if already initialized"
  dihes=(int(*)[4])calloc(diheCount,sizeof(int[4]));
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<diheCount; i++) {
    for (j=0; j<4; j++) {
      dihes[i][j]=io_nexti(line,fp,"psf dihedral atom")-1;
      if (dihes[i][j]>=atomCount || dihes[i][j]<0) {
        fatal(__FILE__,__LINE__,"Atom %d in dihedral %d is out of range\n",dihes[i][j],i);
      }
    }
  }
  
  // Read imprs
  fgets(line, MAXLENGTHSTRING, fp);
  imprCount=io_nexti(line,fp,"psf number of impropers");
#warning "Memory leak if already initialized"
  imprs=(int(*)[4])calloc(imprCount,sizeof(int[4]));
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<imprCount; i++) {
    for (j=0; j<4; j++) {
      imprs[i][j]=io_nexti(line,fp,"psf improper dih atom")-1;
      if (imprs[i][j]>=atomCount || imprs[i][j]<0) {
        fatal(__FILE__,__LINE__,"Atom %d in improper dih %d is out of range\n",imprs[i][j],i);
      }
    }
  }
  
  // Ignore donors
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf number of donors");
  for (i=0; i<2*j; i++) {
    io_nexti(line,fp,"psf donor atom");
  }
  
  // Ignore acceptors
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf number of acceptors");
  for (i=0; i<2*j; i++) {
    io_nexti(line,fp,"psf acceptor atom");
  }
  
  // Not even sure what this section is...
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf nnb???");
  for (i=0; i<atomCount; i++) {
    io_nexti(line,fp,"psf nnb???");
  }

  // Or this one...
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf ngrp???");
  for (i=0; i<3*j; i++) {
    io_nexti(line,fp,"psf ngrp???");
  }

  // OR this one...
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf molnt???");
  for (i=0; i<atomCount; i++) {
    io_nexti(line,fp,"psf molnt???");
  }

  // ignore lone pairs
  fgets(line, MAXLENGTHSTRING, fp);
  i=io_nexti(line,fp,"psf lone pairs");
  j=io_nexti(line,fp,"psf lone pair hydrogrens");
  if (i!=0 || j!=0) {
    fatal(__FILE__,__LINE__,"Program is not set up to treat lone pairs. Found NUMLP=%d NUMLPH=%d in psf\n",i,j);
  }
  
  // Read cmaps
  fgets(line, MAXLENGTHSTRING, fp);
  cmapCount=io_nexti(line,fp,"psf number of cmaps");
#warning "Memory leak if already initialized"
  cmaps=(int(*)[8])calloc(cmapCount,sizeof(int[8]));
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<cmapCount; i++) {
    for (j=0; j<8; j++) {
      cmaps[i][j]=io_nexti(line,fp,"psf cmap atom")-1;
      if (cmaps[i][j]>=atomCount || cmaps[i][j]<0) {
        fatal(__FILE__,__LINE__,"Atom %d in cmap %d is out of range\n",cmaps[i][j],i);
      }
    }
  }
}

void Structure::dump()
{
  fprintf(stdout,"%s:%d IMPLEMENT Structure::dump function.\n",__FILE__,__LINE__);
}
