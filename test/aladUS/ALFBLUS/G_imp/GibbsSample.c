#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double randDouble()
{
  return (rand()+0.5)/(RAND_MAX+1.0);
}

int main(int argc, char *argv[])
{
  int Ns;
  int Nl=400;
  int Nl2=20;
  int Neq=20000;
  int Nmc=2000000;
  double *theta,*lambda;
  long int *p1,*p12,*p2;
  int i,j,k,l,m;
  double b, s, norm, nHigh;
  double thetaNew,eOld,eNew;
  FILE *fp;
  char fnm[128];

  if (argc==2) {
    i=sscanf(argv[1],"%d",&Ns);
    if (i!=1) {
      fprintf(stderr,"Error converting argument to integer\n");
      exit(1);
    }
  } else {
    fprintf(stderr,"Error, function needs exactly 1 argument\n");
    exit(1);
  }
  theta=(double*)calloc(Ns,sizeof(double));
  lambda=(double*)calloc(Ns,sizeof(double));
  p1=(long int*)calloc(Nl,sizeof(long int));
  p12=(long int*)calloc(Nl2*Nl2,sizeof(long int));
  p2=(long int*)calloc(Nl,sizeof(long int));

  b=1;
  for (i=0; i<50; i++) {
    b=0.5*log(0.25*b*Ns*Ns*M_PI/2);
    if (!(b>0)) b=0;
  }

  theta[0]=M_PI/2;
  for (i=1; i<Ns; i++) {
    theta[i]=3*M_PI/2;
  }

  for (i=0; i<Neq+Nmc; i++) {
    if (i%Neq==0) {
      fprintf(stdout,"Step %d\n",i);
    }

    for (j=0; j<Ns; j++) {
      s=(-0.5*sin(theta[j])+0.5);
      eOld=-b*s*s*s*s;

      thetaNew=2*M_PI*randDouble();
      s=(-0.5*sin(thetaNew)+0.5);
      eNew=-b*s*s*s*s;

      if (exp(eOld-eNew)>randDouble()) {
        theta[j]=thetaNew;
      }
    }

    if (i>Neq) {
      norm=0;
      for (j=0; j<Ns; j++) {
        norm+=exp(5.5*sin(theta[j]));
      }
      for (j=0; j<Ns; j++) {
        lambda[j]=exp(5.5*sin(theta[j]))/norm;
      }
      for (j=0; j<Ns; j++) {
        l=(int)floor(Nl*lambda[j]);
        p1[l]++;
        l=(int)floor(Nl2*lambda[j]);
        for (k=j+1; k<Ns; k++) {
          if (lambda[j]+lambda[k]>0.8) {
            m=(int)floor(Nl*lambda[j]/(lambda[j]+lambda[k]));
            p12[m]++;
          }
          m=(int)floor(Nl2*lambda[k]);
          p2[l*Nl2+m]++;
        }
      }
    }
  }

  sprintf(fnm,"G1_%d.dat",Ns);
  fp=fopen(fnm,"w");
  for (i=0; i<Nl; i++) {
    fprintf(fp,"%g\n",-log(p1[i]));
  }
  fclose(fp);

  sprintf(fnm,"G12_%d.dat",Ns);
  fp=fopen(fnm,"w");
  for (i=0; i<Nl; i++) {
    fprintf(fp,"%g\n",-log(p12[i]+p12[Nl-1-i]));
  }
  fclose(fp);

  sprintf(fnm,"G2_%d.dat",Ns);
  fp=fopen(fnm,"w");
  for (i=0; i<Nl2; i++) {
    for (j=0; j<Nl2; j++) {
      fprintf(fp," %g",-log(p2[i*Nl2+j]+p2[j*Nl2+i]));
    }
    fprintf(fp,"\n");
  }
  fclose(fp);

  sprintf(fnm,"FPL_%d.dat",Ns);
  fp=fopen(fnm,"w");
  norm=0;
  nHigh=0;
  for (i=0; i<Nl; i++) {
    norm+=p1[i];
    if (i>=((99*Nl)/100)) {
      nHigh+=p1[i];
    }
  }
  fprintf(fp,"%g\n",Ns*nHigh/norm);
  fclose(fp);
}
