
#include "topol.h"

#include "defines.h"

#include "md.h"
#include "parms.h"
#include "state.h"
#include "sort.h"
#include "error.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define KCALORIE 4.186
#define REALLOC_STRIDE 128


static
void checkatom(int i,int N,char* line)
{
  if (i-1<0 || i-1>=N) {
    char message[MAXLENGTH];
    sprintf(message,"atom %d in top file does not exist\n%s\n",i,line);
    fatalerror(message);
  }
}


static
void topseek(const char directive[],FILE *fp)
{
  char s[MAXLENGTH];
  if (fp==NULL) {
    char message[MAXLENGTH];
    sprintf(message,"top file does not exist\n");
    fatalerror(message);
  }
  while (fgets(s,MAXLENGTH,fp) != NULL) {
    if ((strcmp(s,directive) == 0) || (strcmp(s+1,directive) == 0)) {
      break;
    }
  }
}


static
int topbond(int bSave,int Natom,struct_bondparms *bondparms,FILE *fp)
{
  char s[MAXLENGTH];
  int N=0;
  int i,j;
  double k0_d,r0_d;
  real k0,r0;
  int type;

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    if (s[0] == ';' || s[1] == ';') {
      // goto next line
      ;
    } else if (sscanf(s,"%d %d %d %lg %lg",&i,&j,&type,&r0_d,&k0_d)>=3) {
      k0=k0_d;
      r0=r0_d;
      if (bSave) {
        if (type==1) {
          // great
        } else if (type==5) {
          k0=0;
          r0=0;
        } else {
          fprintf(stderr,"Warning, unrecognized bond type %d\n",type);
        }
        checkatom(i,Natom,s);
        checkatom(j,Natom,s);
        bondparms[N].i=i-1;
        bondparms[N].j=j-1;
        bondparms[N].k0=k0;
        bondparms[N].r0=r0;
      }
      N++;
    } else {
      break;
    }
  }
  return N;
}


static
struct_bondparms* alloc_bondparms(int *N,int Natom,char fnm[])
{
  struct_bondparms *tmp, *bondparms;
  FILE *fp;
  // int i;

  fp=fopen(fnm,"r");
  topseek("[ bonds ]\n",fp);
  *N=topbond(0,Natom,NULL,fp);
  fclose(fp);

  tmp=(struct_bondparms*) calloc(*N,sizeof(struct_bondparms));
  cudaMalloc(&bondparms,*N*sizeof(struct_bondparms));

  fp=fopen(fnm,"r");
  topseek("[ bonds ]\n",fp);
  *N=topbond(1,Natom,tmp,fp);
  fclose(fp);

  cudaMemcpy(bondparms,tmp,*N*sizeof(struct_bondparms),cudaMemcpyHostToDevice);
  free(tmp);

  return bondparms;
}


static
int topangle(int bSave,int Natom,struct_angleparms *angleparms,FILE *fp)
{
  char s[MAXLENGTH];
  int N=0;
  int i,j,k;
  double k0_d,t0_d;
  real k0,t0;

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    if (s[0] == ';' || s[1] == ';') {
      // goto next line
      ;
    } else if (sscanf(s,"%d %d %d 1 %lg %lg",&i,&j,&k,&t0_d,&k0_d)==5) {
      k0=k0_d;
      t0=t0_d;
      if (bSave) {
        checkatom(i,Natom,s);
        checkatom(j,Natom,s);
        checkatom(k,Natom,s);
        angleparms[N].i=i-1;
        angleparms[N].j=j-1;
        angleparms[N].k=k-1;
        angleparms[N].k0=k0;
        angleparms[N].t0=(M_PI/180)*t0;
      }
      N++;
    } else {
      break;
    }
  }
  return N;
}


static
struct_angleparms* alloc_angleparms(int *N,int Natom,char fnm[])
{
  struct_angleparms *tmp, *angleparms;
  FILE *fp;
  // int i;

  fp=fopen(fnm,"r");
  topseek("[ angles ]\n",fp);
  *N=topangle(0,Natom,NULL,fp);
  fclose(fp);

  tmp=(struct_angleparms*) calloc(*N,sizeof(struct_angleparms));
  cudaMalloc(&angleparms,*N*sizeof(struct_angleparms));

  fp=fopen(fnm,"r");
  topseek("[ angles ]\n",fp);
  *N=topangle(1,Natom,tmp,fp);
  fclose(fp);

  cudaMemcpy(angleparms,tmp,*N*sizeof(struct_angleparms),cudaMemcpyHostToDevice);
  free(tmp);

  return angleparms;
}


static
int topdih(int bSave,int Natom,struct_dihparms *dihparms,FILE *fp)
{
  char s[MAXLENGTH];
  int N=0;
  int i,j,k,l,ty,n;
  int ip=0;
  int jp=0;
  int kp=0;
  int lp=0;
  double k0_d,p0_d;
  real k0,p0;

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    if (s[0] == ';' || s[1] == ';') {
      // goto next line
      ;
    } else if (sscanf(s,"%d %d %d %d %d %lg %lg %d",
                         &i,&j,&k,&l,&ty,&p0_d,&k0_d,&n)>=7) {
      k0=k0_d;
      p0=p0_d;
      if (ty==1) {
       // if (i==ip && j==jp && k==kp && l==lp && dihparms[N-1].n2==0) {
       if (i==ip && j==jp && k==kp && l==lp) {
        if (bSave) {
          dihparms[N-1].k02=k0;
          dihparms[N-1].p02=(M_PI/180)*p0;
          dihparms[N-1].n2=(real) n;
        }
        ip=0; jp=0; kp=0; lp=0; // Reset the previous indices so a third won't be added
        // No N++;
       } else {
        if (bSave) {
          checkatom(i,Natom,s);
          checkatom(j,Natom,s);
          checkatom(k,Natom,s);
          checkatom(l,Natom,s);
          dihparms[N].i=i-1;
          dihparms[N].j=j-1;
          dihparms[N].k=k-1;
          dihparms[N].l=l-1;
          dihparms[N].k0=k0;
          dihparms[N].p0=(M_PI/180)*p0;
          dihparms[N].n=(real) n;
          dihparms[N].n2=0;
        }
        ip=i; jp=j; kp=k; lp=l; // Set previous indices so a second can be added
        N++;
       }
      } else if (ty==2) {
        if (bSave) {
          checkatom(i,Natom,s);
          checkatom(j,Natom,s);
          checkatom(k,Natom,s);
          checkatom(l,Natom,s);
          dihparms[N].i=i-1;
          dihparms[N].j=j-1;
          dihparms[N].k=k-1;
          dihparms[N].l=l-1;
          dihparms[N].k0=k0;
          dihparms[N].p0=(M_PI/180)*p0;
          dihparms[N].n=0;
          dihparms[N].n2=0;
        }
        ip=0; jp=0; kp=0; lp=0; // Reset previous indices
        N++;
      } else {
        fprintf(stderr,"Unrecognized Dihedral Type\n");
        N++;
      }
    } else {
      break;
    }
  }
  return N;
}


static
struct_dihparms* alloc_dihparms(int *N,int Natom,char fnm[])
{
  struct_dihparms *tmp, *dihparms;
  FILE *fp;

  fp=fopen(fnm,"r");
  topseek("[ dihedrals ]\n",fp);
  *N=topdih(0,Natom,NULL,fp);
  fclose(fp);

  tmp=(struct_dihparms*) calloc(*N,sizeof(struct_dihparms));
  cudaMalloc(&dihparms,*N*sizeof(struct_dihparms));

  fp=fopen(fnm,"r");
  topseek("[ dihedrals ]\n",fp);
  *N=topdih(1,Natom,tmp,fp);
  fclose(fp);

  cudaMemcpy(dihparms,tmp,*N*sizeof(struct_dihparms),cudaMemcpyHostToDevice);
  free(tmp);

  return dihparms;
}


static
int toppair1(int bSave,int Natom,struct_pair1parms *pair1parms,FILE *fp)
{
  char s[MAXLENGTH];
  int N=0;
  int i,j,ty;
  double C6_d,C12_d;
  real C6,C12;

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    if (s[0] == ';' || s[1] == ';') {
      // goto next line
      ;
    } else if (sscanf(s,"%d %d %d %lg %lg",&i,&j,&ty,&C6_d,&C12_d)>=5) {
      C6=C6_d;
      C12=C12_d;
      if (bSave) {
        if (ty==1) {
          checkatom(i,Natom,s);
          checkatom(j,Natom,s);
          pair1parms[N].i=i-1;
          pair1parms[N].j=j-1;
          pair1parms[N].C6=C6;
          pair1parms[N].C12=C12;
        } else if (ty!=5 && ty!=6 && ty!=7 && ty!=8) {
          fprintf(stderr,"Unrecognized pair type\n");
        }
      }
      if (ty==1) {
        N++;
      }
    } else {
      break;
    }
  }
  return N;
}


static
struct_pair1parms* alloc_pair1parms(int *N,int Natom,char fnm[])
{
  struct_pair1parms *tmp, *pair1parms;
  FILE *fp;
  // int i;

  fp=fopen(fnm,"r");
  topseek("[ pairs ]\n",fp);
  *N=toppair1(0,Natom,NULL,fp);
  fclose(fp);

  pair1parms=(struct_pair1parms*) calloc(*N,sizeof(struct_pair1parms));

  tmp=(struct_pair1parms*) calloc(*N,sizeof(struct_pair1parms));
  cudaMalloc(&pair1parms,*N*sizeof(struct_pair1parms));

  fp=fopen(fnm,"r");
  topseek("[ pairs ]\n",fp);
  *N=toppair1(1,Natom,tmp,fp);
  fclose(fp);

  cudaMemcpy(pair1parms,tmp,*N*sizeof(struct_pair1parms),cudaMemcpyHostToDevice);
  free(tmp);

  return pair1parms;
}


// bare Gaussian potential:
// V_ij = -A exp( - ( r - mu )^2 / ( 2 sigma^2 ) )
// selected in the [ pairs ] section;
// ; i j ftype A mu sigma
// with ftype = 5
// note that the amplitude of the Gaussian is -A.
// This can be used with a separate repulsion term or without it.
static
int toppair5(int bSave,int Natom,struct_pair5parms *pair5parms,FILE *fp)
{
  char s[MAXLENGTH];
  int N=0;
  int i,j,ty;
  double eps_d,r0_d,sigma_d;
  real eps,r0,sigma;

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    if (s[0] == ';' || s[1] == ';') {
      // goto next line
      ;
    } else if (sscanf(s,"%d %d %d %lg %lg %lg",&i,&j,&ty,&eps_d,&r0_d,&sigma_d)>=5) {
      eps=eps_d;
      r0=r0_d;
      sigma=sigma_d;
      if (bSave) {
        if (ty==5) {
          checkatom(i,Natom,s);
          checkatom(j,Natom,s);
          pair5parms[N].i=i-1;
          pair5parms[N].j=j-1;
          pair5parms[N].eps=eps;
          pair5parms[N].r0=r0;
          pair5parms[N].sigma=sigma;
        } else if (ty!=1 && ty!=6 && ty!=7 && ty!=8) {
          fprintf(stderr,"Unrecognized pair type\n");
        }
      }
      if (ty==5) {
        N++;
      }
    } else {
      break;
    }
  }
  return N;
}


static
struct_pair5parms* alloc_pair5parms(int *N,int Natom,const char fnm[])
{
  struct_pair5parms *tmp, *pair5parms;
  FILE *fp;
  // int i;

  fp=fopen(fnm,"r");
  topseek("[ pairs ]\n",fp);
  *N=toppair5(0,Natom,NULL,fp);
  fclose(fp);

  pair5parms=(struct_pair5parms*) calloc(*N,sizeof(struct_pair5parms));

  tmp=(struct_pair5parms*) calloc(*N,sizeof(struct_pair5parms));
  cudaMalloc(&pair5parms,*N*sizeof(struct_pair5parms));

  fp=fopen(fnm,"r");
  topseek("[ pairs ]\n",fp);
  *N=toppair5(1,Natom,tmp,fp);
  fclose(fp);

  cudaMemcpy(pair5parms,tmp,*N*sizeof(struct_pair5parms),cudaMemcpyHostToDevice);
  free(tmp);

  return pair5parms;
}


static
int toppair6(int bSave,int Natom,struct_pair6parms *pair6parms,FILE *fp)
{
  char s[MAXLENGTH];
  int N=0;
  int i,j,ty;
  double eps_d,r0_d,sigma_d,eps12_d;
  real eps,r0,sigma,eps12;

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    if (s[0] == ';' || s[1] == ';') {
      // goto next line
      ;
    } else if (sscanf(s,"%d %d %d %lg %lg %lg %lg",&i,&j,&ty,&eps_d,&r0_d,&sigma_d,&eps12_d)>=5) {
      eps=eps_d;
      r0=r0_d;
      sigma=sigma_d;
      eps12=eps12_d;
      if (bSave) {
        if (ty==6) {
          checkatom(i,Natom,s);
          checkatom(j,Natom,s);
          pair6parms[N].i=i-1;
          pair6parms[N].j=j-1;
          pair6parms[N].eps=eps;
          pair6parms[N].r0=r0;
          pair6parms[N].sigma=sigma;
          pair6parms[N].eps12=eps12;
        } else if (ty!=1 && ty!=5 && ty!=7 && ty!=8) {
          fprintf(stderr,"Unrecognized pair type\n");
        }
      }
      if (ty==6) {
        N++;
      }
    } else {
      break;
    }
  }
  return N;
}


static
struct_pair6parms* alloc_pair6parms(int *N,int Natom,char fnm[])
{
  struct_pair6parms *tmp, *pair6parms;
  FILE *fp;
  // int i;

  fp=fopen(fnm,"r");
  topseek("[ pairs ]\n",fp);
  *N=toppair6(0,Natom,NULL,fp);
  fclose(fp);

  pair6parms=(struct_pair6parms*) calloc(*N,sizeof(struct_pair6parms));

  tmp=(struct_pair6parms*) calloc(*N,sizeof(struct_pair6parms));
  cudaMalloc(&pair6parms,*N*sizeof(struct_pair6parms));

  fp=fopen(fnm,"r");
  topseek("[ pairs ]\n",fp);
  *N=toppair6(1,Natom,tmp,fp);
  fclose(fp);

  cudaMemcpy(pair6parms,tmp,*N*sizeof(struct_pair6parms),cudaMemcpyHostToDevice);
  free(tmp);

  return pair6parms;
}


static
int toppair7(int bSave,int Natom,struct_pair7parms *pair7parms,FILE *fp)
{
  char s[MAXLENGTH];
  int N=0;
  int i,j,ty;
  double eps_d,r01_d,sigma1_d,r02_d,sigma2_d,eps12_d;
  real eps,r01,sigma1,r02,sigma2,eps12;

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    if (s[0] == ';' || s[1] == ';') {
      // goto next line
      ;
    } else if (sscanf(s,"%d %d %d %lg %lg %lg %lg %lg %lg", \
                      &i,&j,&ty,&eps_d,&r01_d,&sigma1_d,&r02_d,&sigma2_d,&eps12_d)>=5) {
      eps=eps_d;
      r01=r01_d;
      sigma1=sigma1_d;
      r02=r02_d;
      sigma2=sigma2_d;
      eps12=eps12_d;
      if (bSave) {
        if (ty==7) {
          checkatom(i,Natom,s);
          checkatom(j,Natom,s);
          pair7parms[N].i=i-1;
          pair7parms[N].j=j-1;
          pair7parms[N].eps=eps;
          pair7parms[N].r01=r01;
          pair7parms[N].sigma1=sigma1;
          pair7parms[N].r02=r02;
          pair7parms[N].sigma2=sigma2;
          pair7parms[N].eps12=eps12;
        } else if (ty!=1 && ty!=5 && ty!=6 && ty!=8) {
          fprintf(stderr,"Unrecognized pair type\n");
        }
      }
      if (ty==7) {
        N++;
      }
    } else {
      break;
    }
  }
  return N;
}


static
struct_pair7parms* alloc_pair7parms(int *N,int Natom,const char fnm[])
{
  struct_pair7parms *tmp, *pair7parms;
  FILE *fp;
  // int i;

  fp=fopen(fnm,"r");
  topseek("[ pairs ]\n",fp);
  *N=toppair7(0,Natom,NULL,fp);
  fclose(fp);

  pair7parms=(struct_pair7parms*) calloc(*N,sizeof(struct_pair7parms));

  tmp=(struct_pair7parms*) calloc(*N,sizeof(struct_pair7parms));
  cudaMalloc(&pair7parms,*N*sizeof(struct_pair7parms));

  fp=fopen(fnm,"r");
  topseek("[ pairs ]\n",fp);
  *N=toppair7(1,Natom,tmp,fp);
  fclose(fp);

  cudaMemcpy(pair7parms,tmp,*N*sizeof(struct_pair7parms),cudaMemcpyHostToDevice);
  free(tmp);

  return pair7parms;
}


static
int toppair8(int bSave,int Natom,struct_pair8parms *pair8parms,FILE *fp)
{
  char s[MAXLENGTH];
  int N=0;
  int i,j,ty;
  double amp_d,mu_d,sigma_d;
  real amp,mu,sigma;

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    if (s[0] == ';' || s[1] == ';') {
      // goto next line
      ;
    } else if (sscanf(s,"%d %d %d %lg %lg %lg", \
                      &i,&j,&ty,&amp_d,&mu_d,&sigma_d)>=5) {
      amp=amp_d;
      mu=mu_d;
      sigma=sigma_d;
      if (bSave) {
        if (ty==8) {
          checkatom(i,Natom,s);
          checkatom(j,Natom,s);
          pair8parms[N].i=i-1;
          pair8parms[N].j=j-1;
          pair8parms[N].amp=amp;
          pair8parms[N].mu=mu;
          pair8parms[N].sigma=sigma;
        } else if (ty!=1 && ty!=5 && ty!=6 && ty!=7) {
          fprintf(stderr,"Unrecognized pair type\n");
        }
      }
      if (ty==8) {
        N++;
      }
    } else {
      break;
    }
  }
  return N;
}


static
struct_pair8parms* alloc_pair8parms(int *N,int Natom,const char fnm[])
{
  struct_pair8parms *tmp, *pair8parms;
  FILE *fp;
  // int i;

  fp=fopen(fnm,"r");
  topseek("[ pairs ]\n",fp);
  *N=toppair8(0,Natom,NULL,fp);
  fclose(fp);

  pair8parms=(struct_pair8parms*) calloc(*N,sizeof(struct_pair8parms));

  tmp=(struct_pair8parms*) calloc(*N,sizeof(struct_pair8parms));
  cudaMalloc(&pair8parms,*N*sizeof(struct_pair8parms));

  fp=fopen(fnm,"r");
  topseek("[ pairs ]\n",fp);
  *N=toppair8(1,Natom,tmp,fp);
  fclose(fp);

  cudaMemcpy(pair8parms,tmp,*N*sizeof(struct_pair8parms),cudaMemcpyHostToDevice);
  free(tmp);

  return pair8parms;
}


struct_umbrella* alloc_umbrella(struct_md *md)
  {
  int i;
  struct_umbrella *umbrella = NULL;
  FILE *umbrella_file = NULL;
  double k_umb_d,Q_ref_d,Q_start_d,Q_steps_d;
  umbrella_file = fopen("umbrella_params","r");

  if ( umbrella_file ) {
    umbrella = (struct_umbrella*) malloc(sizeof(struct_umbrella));
    fscanf(umbrella_file,"freq_out %d\n",&umbrella->freq_out);
    for (i=0; i<=md->parms->id; i++) {
      fscanf(umbrella_file,"%lf %lf %lf %lf",&k_umb_d,&Q_ref_d,&Q_start_d,&Q_steps_d);
      umbrella->k_umb=k_umb_d;
      umbrella->Q_ref=Q_ref_d;
      umbrella->Q_start=Q_start_d;
      umbrella->Q_steps=Q_steps_d;
      umbrella->type=md->parms->umbrella_function;
    }
    cudaMalloc(&(umbrella->Q),sizeof(real));
    cudaMalloc(&(umbrella->EQ),sizeof(real));
    fclose(umbrella_file);
    
  }

  return umbrella;
}


void free_umbrella(struct_umbrella* umbrella)
{
  if (umbrella) {
    cudaFree(umbrella->Q);
    cudaFree(umbrella->EQ);
    free(umbrella);
  }
}


static
void addexcl(int in,struct_exclparms *exclparms,int i,int j)
{
  int ij;

  // See if we already have j in i's list
  for (ij=0; ij<exclparms[i].N[in]; ij++) {
    if (exclparms[i].j[ij]==j) {
      return;
    }
  }

  // If not, add j to i's list
  if (exclparms[i].N[in]==exclparms[i].Nmax) {
    exclparms[i].Nmax+=5;
    exclparms[i].j=(int*) realloc(exclparms[i].j,exclparms[i].Nmax*sizeof(int));
  }
  exclparms[i].j[exclparms[i].N[in]]=j;
  exclparms[i].N[in]++;
}

/*
static
int topcount(FILE *fp)
{
  char s[MAXLENGTH];
  int i,N;

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    if (s[0] == ';' || s[1] == ';') {
      // goto next line
      ;
    } else if (sscanf(s,"%d",&i)==1) {
      N=i;
    } else {
      break;
    }
  }
  return N;
}
*/

static
void topexcl(int in,int Natom,struct_exclparms *exclparms,FILE *fp)
{
  char s[MAXLENGTH];
  int i,j;

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    if (s[0] == ';' || s[1] == ';') {
      // goto next line
      ;
    } else if (sscanf(s,"%d %d",&i,&j)==2) {
      checkatom(i,Natom,s);
      checkatom(j,Natom,s);
      addexcl(in,exclparms,i-1,j-1);
      addexcl(in,exclparms,j-1,i-1);
    } else {
      break;
    }
  }
}


static
void updateexcl(int in,int N,struct_exclparms *exclparms)
{
  int i,j,k,j1,j2;

  for (i=0; i<N; i++) {
    exclparms[i].N[in]=exclparms[i].N[in-1];
    for (j=0; j<exclparms[i].N[in-1]; j++) {
      j1=exclparms[i].j[j];
      for (k=0; k<exclparms[j1].N[0]; k++) {
        j2=exclparms[j1].j[k];
        addexcl(in,exclparms,i,j2);
      }
    }
  }
}


static
struct_exclparms* alloc_exclparms(int *N,int Natom,const char fnm[])
{
  struct_exclparms *exclparms;
  FILE *fp;
  // int i;

  // Allocate for each atom
  exclparms=(struct_exclparms*) calloc(*N,sizeof(struct_exclparms));
  
  fp=fopen(fnm,"r");
  topseek("[ bonds ]\n",fp);
  topexcl(0,Natom,exclparms,fp);
  fclose(fp);

  updateexcl(1,*N,exclparms);
  updateexcl(2,*N,exclparms);

  fp=fopen(fnm,"r");
  topseek("[ exclusions ]\n",fp);
  topexcl(2,Natom,exclparms,fp);
  fclose(fp);

  return exclparms;
}

/*
static
struct_exclparms* alloc_exclparmsdev(int N,struct_exclparms* excl)
{
  struct_exclparms *exclmirror, *excldev;
  int i;

  exclmirror=(struct_exclparms*) malloc(N*sizeof(struct_exclparms));
  cudaMalloc(&excldev,N*sizeof(struct_exclparms));

  for (i=0; i<N; i++) {
    exclmirror[i].N[0]=excl[i].N[0];
    exclmirror[i].N[1]=excl[i].N[1];
    exclmirror[i].N[2]=excl[i].N[2];
    exclmirror[i].Nmax=excl[i].Nmax;
    if (exclmirror[i].Nmax) {
      cudaMalloc(&(exclmirror[i].j),exclmirror[i].Nmax*sizeof(int));
      cudaMemcpy(exclmirror[i].j,excl[i].j,exclmirror[i].Nmax*sizeof(int),cudaMemcpyHostToDevice);
    }
  }

  cudaMemcpy(excldev,exclmirror,N*sizeof(struct_exclparms),cudaMemcpyHostToDevice);

  free(exclmirror);

  return excldev;
}


static
void free_exclparmsdev(int N,struct_exclparms* excldev)
{
  struct_exclparms *exclmirror;
  int i;

  exclmirror=(struct_exclparms*) malloc(N*sizeof(struct_exclparms));
  cudaMemcpy(exclmirror,excldev,N*sizeof(struct_exclparms),cudaMemcpyDeviceToHost);

  for (i=0; i<N; i++) {
    if (exclmirror[i].Nmax) {
      cudaFree(exclmirror[i].j);
    }
  }

  free(exclmirror);
  cudaFree(excldev);
}
*/

static
struct_exclhash* alloc_exclhash(int N)
{
  struct_exclhash *exclhash;

  exclhash=(struct_exclhash*) malloc(sizeof(struct_exclhash));
  exclhash->Nind=XHASH*N+1;
  exclhash->ind=(int*) malloc(exclhash->Nind*sizeof(int));
  exclhash->Ndata=exclhash->Nind;
  exclhash->data=(int*) malloc(exclhash->Ndata*sizeof(int));

  exclhash->ind[0]=0;

  return exclhash;
}


static
void realloc_exclhash(struct_exclhash* exclhash)
{
  exclhash->Ndata+=exclhash->Nind;
  exclhash->data=(int*) realloc(exclhash->data,exclhash->Ndata*sizeof(int));
}


static
void free_exclhash(struct_exclhash* exclhash)
{
  free(exclhash->ind);
  free(exclhash->data);
  free(exclhash);
}


static
struct_exclhash* make_exclhash(int N,struct_exclparms* exclparms)
{
  struct_exclhash *exclhash;
  int i,j;
  int ind, hashval, jind;

  exclhash=alloc_exclhash(N);
  ind=0;

  for (i=0; i<N; i++) {
    for (hashval=0; hashval<XHASH; hashval++) {
      for (jind=0; jind<exclparms[i].N[2]; jind++) {
        j=exclparms[i].j[jind];
        if ((j % XHASH)==hashval) {
          if (ind>=exclhash->Ndata) {
            realloc_exclhash(exclhash);
          }
          exclhash->data[ind]=j;
          ind++;
        }
      }
      exclhash->ind[i*XHASH+hashval+1]=ind;
    }
  }

  return exclhash;
}


struct_exclhash* make_empty_exclhash(int N)
{
  struct_exclhash *exclhash;
  int i;
  int hashval;

  exclhash=alloc_exclhash(N);

  for (i=0; i<N; i++) {
    for (hashval=0; hashval<XHASH; hashval++) {
      exclhash->ind[i*XHASH+hashval+1]=0;
    }
  }

  return exclhash;
}


struct_exclhash* alloc_exclhashdev(struct_exclhash* exclhash)
{
  struct_exclhash *exclhashdev;

  exclhashdev=(struct_exclhash*) malloc(sizeof(struct_exclhash));

  exclhashdev->Nind=exclhash->Nind;
  cudaMalloc(&(exclhashdev->ind),exclhash->Nind*sizeof(int));
  cudaMemcpy(exclhashdev->ind,exclhash->ind,exclhash->Nind*sizeof(int),cudaMemcpyHostToDevice);

  exclhashdev->Ndata=exclhash->Ndata;
  cudaMalloc(&(exclhashdev->data),exclhash->Ndata*sizeof(int));
  cudaMemcpy(exclhashdev->data,exclhash->data,exclhash->Ndata*sizeof(int),cudaMemcpyHostToDevice);

  return exclhashdev;
}


static
void free_exclhashdev(struct_exclhash* exclhashdev)
{
  cudaFree(exclhashdev->ind);
  cudaFree(exclhashdev->data);
  free(exclhashdev);
}


static
void parsebond_kbgo(char *s,int Natom,int *N,int *mN,struct_bondparms **bondparms)
{
  int i,j;
  double k0_d,r0_d;
  real k0,r0;

  if (sscanf(s,"G%d G%d %lg %lg",&i,&j,&k0_d,&r0_d)==4) {
    r0=r0_d;
    k0=k0_d;
    if (N[0]>=mN[0]) {
      mN[0]+=REALLOC_STRIDE;
      bondparms[0]=(struct_bondparms*) realloc(bondparms[0],mN[0]*sizeof(struct_bondparms));
    }
    checkatom(i,Natom,s);
    checkatom(j,Natom,s);
    bondparms[0][N[0]].i=i-1;
    bondparms[0][N[0]].j=j-1;
    bondparms[0][N[0]].k0=2.0*k0*KCALORIE*100.0;
    bondparms[0][N[0]].r0=r0/10.0;
    N[0]++;
  } else {
    fprintf(stderr,"Unread: %s",s);
  }
}


static
void parseangle_kbgo(char *s,int Natom,int *N,int *mN,struct_angleparms **angleparms)
{
  int i,j,k;
  double k0_d,t0_d;
  real k0,t0;

  if (sscanf(s,"G%d G%d G%d %lg %lg",&i,&j,&k,&k0_d,&t0_d)==5) {
    k0=k0_d;
    t0=t0_d;
    if (N[0]>=mN[0]) {
      mN[0]+=REALLOC_STRIDE;
      angleparms[0]=(struct_angleparms*) realloc(angleparms[0],mN[0]*sizeof(struct_angleparms));
    }
    checkatom(i,Natom,s);
    checkatom(j,Natom,s);
    checkatom(k,Natom,s);
    angleparms[0][N[0]].i=i-1;
    angleparms[0][N[0]].j=j-1;
    angleparms[0][N[0]].k=k-1;
    angleparms[0][N[0]].k0=2.0*k0*KCALORIE;
    angleparms[0][N[0]].t0=(M_PI/180)*t0;
    N[0]++;
  } else {
    fprintf(stderr,"Unread: %s",s);
  }
}


static
void parsedih_kbgo(char *s,int Natom,int *N,int *mN,struct_dihparms **dihparms,real *kbgo_dih0)
{
  int i,j,k,l,n;
  int ip,jp,kp,lp,n2p;
  double k0_d,p0_d;
  real k0,p0;

  if (sscanf(s,"G%d G%d G%d G%d %lg %d %lg",&i,&j,&k,&l,&k0_d,&n,&p0_d)==7) {
    k0=k0_d;
    p0=p0_d;
    kbgo_dih0[0]+=k0*KCALORIE;
    if (N[0]>=mN[0]) {
      mN[0]+=REALLOC_STRIDE;
      dihparms[0]=(struct_dihparms*) realloc(dihparms[0],mN[0]*sizeof(struct_dihparms));
    }
    checkatom(i,Natom,s);
    checkatom(j,Natom,s);
    checkatom(k,Natom,s);
    checkatom(l,Natom,s);
    if (N[0]>0) {
      ip=dihparms[0][N[0]-1].i+1;
      jp=dihparms[0][N[0]-1].j+1;
      kp=dihparms[0][N[0]-1].k+1;
      lp=dihparms[0][N[0]-1].l+1;
      n2p=dihparms[0][N[0]-1].n2;
    }
    if (i==ip && j==jp && k==kp && l==lp && n2p==0) {
      dihparms[0][N[0]-1].k02=k0*KCALORIE;
      dihparms[0][N[0]-1].p02=(M_PI/180)*p0;
      dihparms[0][N[0]-1].n2=(real) n;
    } else {
      dihparms[0][N[0]].i=i-1;
      dihparms[0][N[0]].j=j-1;
      dihparms[0][N[0]].k=k-1;
      dihparms[0][N[0]].l=l-1;
      dihparms[0][N[0]].k0=k0*KCALORIE;
      dihparms[0][N[0]].p0=(M_PI/180)*p0;
      dihparms[0][N[0]].n=(real) n;
      dihparms[0][N[0]].n2=0;
      N[0]++;
    }
  } else {
    fprintf(stderr,"Unread: %s",s);
  }
}


static
void parsepair1_kbgo(char *s,int Natom,int *N,int *mN,struct_pair1parms **pair1parms)
{
  int i,j;
  real C6,C12,rmin6;
  double eps_d,rmin_d;
  real eps,rmin;

  if (sscanf(s,"G%d G%d %lg %lg",&i,&j,&eps_d,&rmin_d)==4) {
    eps=eps_d;
    rmin=rmin_d;
    if (N[0]>=mN[0]) {
      mN[0]+=REALLOC_STRIDE;
      pair1parms[0]=(struct_pair1parms*) realloc(pair1parms[0],mN[0]*sizeof(struct_pair1parms));
    }
    checkatom(i,Natom,s);
    checkatom(j,Natom,s);
    pair1parms[0][N[0]].i=i-1;
    pair1parms[0][N[0]].j=j-1;
    rmin6=rmin*rmin*rmin*rmin*rmin*rmin/1e6;
    C6=-2.0*eps*KCALORIE*rmin6;
    C12=-eps*KCALORIE*rmin6*rmin6;
    pair1parms[0][N[0]].C6=C6;
    pair1parms[0][N[0]].C12=C12;
    N[0]++;
  } else {
    fprintf(stderr,"Unread: %s",s);
  }
}


static
void parsenb_kbgo(char *s,int Natom,struct_nbparms *nbparms)
{
  int i;
  // double q_d,eps_d,rmin_d;
  // real q,eps,rmin;
  double eps_d,rmin_d;
  real eps,rmin;

  // if (sscanf(s,"G%d %lg %lg %lg",&i,&q_d,&eps_d,&rmin_d)==4)
  if (sscanf(s,"G%d %*lg %lg %lg",&i,&eps_d,&rmin_d)==3) {
    // q=q_d;
    eps=eps_d;
    rmin=rmin_d;
    checkatom(i,Natom,s);
    nbparms[i-1].eps=-eps*KCALORIE;
    nbparms[i-1].rmin=2.0*(rmin/10.0); // *2.0; // *1.122462048;
  } else {
    fprintf(stderr,"Unread: %s",s);
  }
}


static
struct_exclparms* alloc_exclparms_kbgo(int *N,int N_bond,struct_bondparms *bondparms,int N_pair1,struct_pair1parms *pair1parms)
{
  struct_exclparms *exclparms;
  int i;

  // Allocate for each atom
  exclparms=(struct_exclparms*) calloc(*N,sizeof(struct_exclparms));
  
  for (i=0; i<N_bond; i++) {
    addexcl(0,exclparms,bondparms[i].i,bondparms[i].j);
    addexcl(0,exclparms,bondparms[i].j,bondparms[i].i);
  }

  updateexcl(1,*N,exclparms);
  // updateexcl(2,*N,exclparms);
  for (i=0; i<*N; i++) { // update exclusion 2 to only have angles
    // exclparms[i].N[1]=exclparms[i].N[0];
    exclparms[i].N[2]=exclparms[i].N[1];
  }

  for (i=0; i<N_pair1; i++) {
    addexcl(2,exclparms,pair1parms[i].i,pair1parms[i].j);
    addexcl(2,exclparms,pair1parms[i].j,pair1parms[i].i);
  }

  return exclparms;
}


struct_parms* alloc_topparms_smog(struct_md *md)
{
  // char topfile[]="system/PK_NoNextLocal/PK_NoNextLocal.top";
  // char topfile[]="system/PK/PK.top"; // CONST
  struct_parms *parms=md->parms;
  char *topfile=parms->arg_topfile;
  int Natom=md->state->N_all; // should probably be N_rna

  // fprintf(stderr,"Warning, uncorrected unit mismatch\n");
  parms->bondparms=alloc_bondparms(&(parms->N_bond),Natom,topfile);
  parms->bondblock=alloc_bondblock(parms->bondparms,parms->N_bond);

  parms->angleparms=alloc_angleparms(&(parms->N_angle),Natom,topfile);
  parms->angleblock=alloc_angleblock(parms->angleparms,parms->N_angle);

  parms->dihparms=alloc_dihparms(&(parms->N_dih),Natom,topfile);
  parms->dihblock=alloc_dihblock(parms->dihparms,parms->N_dih);

  parms->pair1parms=alloc_pair1parms(&(parms->N_pair1),Natom,topfile);
  parms->pair5parms=alloc_pair5parms(&(parms->N_pair5),Natom,topfile);
  parms->pair6parms=alloc_pair6parms(&(parms->N_pair6),Natom,topfile);
  parms->pair7parms=alloc_pair7parms(&(parms->N_pair7),Natom,topfile);
  parms->pair8parms=alloc_pair8parms(&(parms->N_pair8),Natom,topfile);
  parms->umbrella=alloc_umbrella(md);
  parms->N_excl=md->state->N_all;
  parms->exclparms=alloc_exclparms(&(parms->N_excl),Natom,topfile);

  return parms;
}


void free_topparms_smog(struct_parms *parms)
{
  int i;
  cudaFree(parms->bondparms);
  free_bondblock(parms->bondblock);

  cudaFree(parms->angleparms);
  free_angleblock(parms->angleblock);

  cudaFree(parms->dihparms);
  free_dihblock(parms->dihblock);

  cudaFree(parms->pair1parms);
  cudaFree(parms->pair5parms);
  cudaFree(parms->pair6parms);
  cudaFree(parms->pair7parms);
  cudaFree(parms->pair8parms);
  free_umbrella(parms->umbrella);
  // fprintf(stderr,"Free exclparms\n");
  for (i=0; i<parms->N_excl; i++) {
    if (parms->exclparms[i].Nmax>0) {
      free(parms->exclparms[i].j);
    }
  }
  free(parms->exclparms);
}


struct_parms* alloc_topparms_kbgo(struct_md *md)
{
  struct_parms *parms=md->parms;
  char *paramfile=parms->arg_paramfile;
  int Natom=md->state->N_all; // should probably be N_rna
  FILE *fp;
  char s[MAXLENGTH];
  int directive=0;
  // int mN_bond,mN_angle,mN_dih,mN_pair1,mN_pair5,mN_pair6,mN_pair7,mN_pair8;
  int mN_bond,mN_angle,mN_dih,mN_pair1;

  parms->kbgo_dih0=0;

  parms->N_bond=0;
  mN_bond=0;
  parms->N_angle=0;
  mN_angle=0;
  parms->N_dih=0;
  mN_dih=0;
  parms->N_pair1=0;
  mN_pair1=0;
  parms->N_pair5=0;
  // mN_pair5=0;
  parms->N_pair6=0;
  // mN_pair6=0;
  parms->N_pair7=0;
  // mN_pair7=0;
  parms->N_pair8=0;
  // mN_pair8=0;
  parms->N_nb=Natom;

  parms->bondparms=NULL;
  parms->angleparms=NULL;
  parms->dihparms=NULL;
  parms->pair1parms=NULL;
  parms->pair5parms=NULL;
  parms->pair6parms=NULL;
  parms->pair7parms=NULL;
  parms->pair8parms=NULL;
  parms->nbparms=(struct_nbparms*) malloc(parms->N_nb*sizeof(struct_nbparms));

  fp=fopen(paramfile,"r");

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    // if ((strcmp(s,"BOND\n") == 0) || (strcmp(s+1,"BOND\n") == 0))
    if (strcmp(s,"BOND\n") == 0) {
      directive=eT_bond;
    } else if (strcmp(s,"ANGLE\n") == 0) {
      directive=eT_angle;
    } else if (strcmp(s,"DIHEDRAL\n") == 0) {
      directive=eT_dih;
    } else if (strcmp(s,"NBFIX\n") == 0) {
      directive=eT_pair1;
    } else if (strncmp(s,"NONBONDED",9) == 0) {
      directive=eT_nb;
    }
    switch(directive) {
      case eT_bond:
        parsebond_kbgo(s,Natom,&parms->N_bond,&mN_bond,&parms->bondparms);
        break;
      case eT_angle:
        parseangle_kbgo(s,Natom,&parms->N_angle,&mN_angle,&parms->angleparms);
        break;
      case eT_dih:
        parsedih_kbgo(s,Natom,&parms->N_dih,&mN_dih,&parms->dihparms,&parms->kbgo_dih0);
        break;
      case eT_pair1:
        parsepair1_kbgo(s,Natom,&parms->N_pair1,&mN_pair1,&parms->pair1parms);
        break;
      case eT_nb:
        parsenb_kbgo(s,Natom,parms->nbparms);
        break;
      default:
        fprintf(stderr,"Unread: %s",s);
        break;
    }
  }

  fclose(fp);

  // Need to start exclusions before bonds and pairs are moved to GPU - finish later
  parms->N_excl=md->state->N_all;
  parms->exclparms=alloc_exclparms_kbgo(&(parms->N_excl),parms->N_bond,parms->bondparms,parms->N_pair1,parms->pair1parms);

  // Move stuff to GPU
  {
    struct_bondparms *tmp_bond;
    tmp_bond=parms->bondparms;
    cudaMalloc(&parms->bondparms,parms->N_bond*sizeof(struct_bondparms));
    cudaMemcpy(parms->bondparms,tmp_bond,parms->N_bond*sizeof(struct_bondparms),cudaMemcpyHostToDevice);
    free(tmp_bond);
  }
  {
    struct_angleparms *tmp_angle;
    tmp_angle=parms->angleparms;
    cudaMalloc(&parms->angleparms,parms->N_angle*sizeof(struct_angleparms));
    cudaMemcpy(parms->angleparms,tmp_angle,parms->N_angle*sizeof(struct_angleparms),cudaMemcpyHostToDevice);
    free(tmp_angle);
  }
  {
    struct_dihparms *tmp_dih;
    tmp_dih=parms->dihparms;
    cudaMalloc(&parms->dihparms,parms->N_dih*sizeof(struct_dihparms));
    cudaMemcpy(parms->dihparms,tmp_dih,parms->N_dih*sizeof(struct_dihparms),cudaMemcpyHostToDevice);
    free(tmp_dih);
  }
  {
    struct_pair1parms *tmp_pair1;
    tmp_pair1=parms->pair1parms;
    cudaMalloc(&parms->pair1parms,parms->N_pair1*sizeof(struct_pair1parms));
    cudaMemcpy(parms->pair1parms,tmp_pair1,parms->N_pair1*sizeof(struct_pair1parms),cudaMemcpyHostToDevice);
    free(tmp_pair1);
  }
  {
    struct_nbparms *tmp_nb;
    tmp_nb=parms->nbparms;
    cudaMalloc(&parms->nbparms,parms->N_nb*sizeof(struct_nbparms));
    cudaMemcpy(parms->nbparms,tmp_nb,parms->N_nb*sizeof(struct_nbparms),cudaMemcpyHostToDevice);
    free(tmp_nb);
  }

  parms->bondblock=alloc_bondblock(parms->bondparms,parms->N_bond);
  parms->angleblock=alloc_angleblock(parms->angleparms,parms->N_angle);
  parms->dihblock=alloc_dihblock(parms->dihparms,parms->N_dih);

  // parms->umbrella=alloc_umbrella(md);
  parms->umbrella=NULL;

  return parms;
}


void free_topparms_kbgo(struct_parms *parms)
{
  int i;
  cudaFree(parms->bondparms);
  free_bondblock(parms->bondblock);

  cudaFree(parms->angleparms);
  free_angleblock(parms->angleblock);

  cudaFree(parms->dihparms);
  free_dihblock(parms->dihblock);

  cudaFree(parms->pair1parms);

  // fprintf(stderr,"Free exclparms\n");
  for (i=0; i<parms->N_excl; i++) {
    if (parms->exclparms[i].Nmax>0) {
      free(parms->exclparms[i].j);
    }
  }
  free(parms->exclparms);
}


struct_parms* alloc_topparms(struct_md *md)
{
  struct_parms *parms=md->parms;

  if (parms->kbgo==0) {
    parms=alloc_topparms_smog(md);
  } else {
    parms=alloc_topparms_kbgo(md);
  }
  parms->exclhash=make_exclhash(parms->N_excl,parms->exclparms);
  parms->exclhashdev=alloc_exclhashdev(parms->exclhash);

  return parms;
}


void free_topparms(struct_parms *parms)
{
  if (parms->kbgo==0) {
    free_topparms_smog(parms);
  } else {
    free_topparms_kbgo(parms);
  }

  free_exclhash(parms->exclhash);
  free_exclhashdev(parms->exclhashdev);
}
