#include "defines.h"

#include "nblist.h"

#include "nbsort.h"
#include "md.h"
#include "state.h"
#include "parms.h"
#include "topol.h"
#include "atoms.h"
#include "times.h"
#include "error.h"

#include <stdio.h>

// H4H is host neighbor list for use on host. H4D is host neighbor list for use on device. D4D is device neighbor list for use on device.
void alloc_nblist(int N,int TN,struct_nblist** H4H,struct_nblist** H4D)
{
  int i,Nb;

  *H4H=(struct_nblist*) malloc(sizeof(struct_nblist));
  *H4D=(struct_nblist*) malloc(sizeof(struct_nblist));
  // cudaMalloc(D4D,sizeof(struct_nblist));

  for (i=N-1, Nb=1; i>0; i/=2, Nb*=2) ;
  if (Nb<BNB) { // prevent mayhem later - wastes some time with extra sorting
    fprintf(stderr,"Warning: could improve neighbor searching for small systems. %s %d\n",__FILE__,__LINE__);
    Nb=2*BNB;
  }

  (*H4H)->N=N;
  (*H4H)->Nbit=Nb;
  (*H4H)->bin=(struct_bin*) calloc(Nb,sizeof(struct_bin));
  // (*H4H)->iblock=(int*) calloc(N,sizeof(int));
  // (*H4H)->Nblock=(int*) calloc(N,sizeof(int));
  (*H4H)->iblock=(int*) calloc(N+1,sizeof(int));
  // (*H4H)->block=(int*) calloc(N,sizeof(int));
  (*H4D)->N=N;
  (*H4D)->Nbit=Nb;
  cudaMalloc(&((*H4D)->bin),Nb*sizeof(struct_bin));
  // cudaMalloc(&((*H4D)->iblock),N*sizeof(int));
  // cudaMalloc(&((*H4D)->Nblock),N*sizeof(int));
  cudaMalloc(&((*H4D)->iblock),(N+1)*sizeof(int));
  // cudaMalloc(&((*H4D)->block),N*sizeof(int));
  // Allocate Tile structure
  (*H4H)->TN=TN;
  (*H4H)->mNtiles=32*(N/16+1);
  // (*H4H)->itile=(int*) malloc((*H4H)->mNtiles*sizeof(int));
  // (*H4H)->jtile=(int*) malloc((*H4H)->mNtiles*sizeof(int));
  (*H4H)->ijtile=(struct_ij*) malloc((*H4H)->mNtiles*sizeof(struct_ij));
  (*H4H)->shift=(vec*) malloc((*H4H)->mNtiles*sizeof(vec));
  // (*H4H)->iB=(int*) malloc((*H4H)->mNtiles*sizeof(int));
  // (*H4H)->jB=(int*) malloc((*H4H)->mNtiles*sizeof(int));
  // (*H4H)->blocal=(int*) malloc((*H4H)->mNtiles*2*sizeof(int));
  // (*H4H)->N_local=(int*) malloc(((*H4H)->mNtiles+BNB/TNB-1)/(BNB/TNB)*sizeof(int));
  (*H4H)->nlist=(unsigned int*) malloc((*H4H)->mNtiles*TN*sizeof(int));
  
  (*H4D)->TN=TN;
  (*H4D)->mNtiles=32*(N/16+1);
  // cudaMalloc(&((*H4D)->itile),(*H4D)->mNtiles*sizeof(int));
  // cudaMalloc(&((*H4D)->jtile),(*H4D)->mNtiles*sizeof(int));
  cudaMalloc(&((*H4D)->ijtile),(*H4D)->mNtiles*sizeof(struct_ij));
  cudaMalloc(&((*H4D)->shift),(*H4D)->mNtiles*sizeof(vec));
  // cudaMalloc(&((*H4D)->iB),(*H4D)->mNtiles*sizeof(int));
  // cudaMalloc(&((*H4D)->jB),(*H4D)->mNtiles*sizeof(int));
  // cudaMalloc(&((*H4D)->blocal),(*H4D)->mNtiles*2*sizeof(int));
  // cudaMalloc(&((*H4D)->N_local),((*H4D)->mNtiles+BNB/TNB-1)/(BNB/TNB)*sizeof(int));
  cudaMalloc(&((*H4D)->nlist),(*H4D)->mNtiles*TN*sizeof(int));

  // fprintf(stderr,"Warning, still allocating debug buffer in %s %d\n",__FILE__,__LINE__);
  // (*H4H)->debug=(int*) malloc(100000*sizeof(int));
  // cudaMalloc(&((*H4D)->debug),100000*sizeof(int));
  // (*H4H)->debug[0]=1;
  // cudaMemcpy((*H4D)->debug,(*H4H)->debug,sizeof(int),cudaMemcpyHostToDevice);
  // cudaMemcpy(*D4D,*H4D,sizeof(struct_nblist),cudaMemcpyHostToDevice);
}


void realloc_nblist(struct_nblist** H4H,struct_nblist** H4D)
{
  int N,TN;
  // cudaMemcpy(*H4D,*D4D,sizeof(struct_nblist),cudaMemcpyDeviceToHost);
  int mNtiles_tmp;
  struct_ij *ijtile_tmp;
  vec *shift_tmp;
  unsigned int *nlist_tmp;

  N=(*H4H)->N;
  TN=(*H4H)->TN;

  (*H4H)->mNtiles+=8*(N/16+1);
  // (*H4H)->itile=(int*) realloc((*H4H)->itile,(*H4H)->mNtiles*sizeof(int));
  // (*H4H)->jtile=(int*) realloc((*H4H)->jtile,(*H4H)->mNtiles*sizeof(int));
  (*H4H)->ijtile=(struct_ij*) realloc((*H4H)->ijtile,(*H4H)->mNtiles*sizeof(struct_ij));
  (*H4H)->shift=(vec*) realloc((*H4H)->shift,(*H4H)->mNtiles*sizeof(vec));
  // (*H4H)->iB=(int*) realloc((*H4H)->iB,(*H4H)->mNtiles*sizeof(int));
  // (*H4H)->jB=(int*) realloc((*H4H)->jB,(*H4H)->mNtiles*sizeof(int));
  // (*H4H)->blocal=(int*) realloc((*H4H)->blocal,(*H4H)->mNtiles*2*sizeof(int));
  // (*H4H)->N_local=(int*) realloc((*H4H)->N_local,((*H4H)->mNtiles+BNB/TNB-1)/(BNB/TNB)*sizeof(int));
  (*H4H)->nlist=(unsigned int*) realloc((*H4H)->nlist,(*H4H)->mNtiles*TN*sizeof(int));

  mNtiles_tmp=(*H4D)->mNtiles;
  (*H4D)->mNtiles+=8*(N/16+1);

  ijtile_tmp=(*H4D)->ijtile;
  cudaMalloc(&((*H4D)->ijtile),(*H4D)->mNtiles*sizeof(struct_ij));
  cudaMemcpy((*H4D)->ijtile,ijtile_tmp,mNtiles_tmp*sizeof(struct_ij),cudaMemcpyDeviceToDevice);
  cudaFree(ijtile_tmp);

  shift_tmp=(*H4D)->shift;
  cudaMalloc(&((*H4D)->shift),(*H4D)->mNtiles*sizeof(vec));
  cudaMemcpy((*H4D)->shift,shift_tmp,mNtiles_tmp*sizeof(vec),cudaMemcpyDeviceToDevice);
  cudaFree(shift_tmp);

  nlist_tmp=(*H4D)->nlist;
  cudaMalloc(&((*H4D)->nlist),(*H4D)->mNtiles*TN*sizeof(int));
  cudaMemcpy((*H4D)->nlist,nlist_tmp,mNtiles_tmp*TN*sizeof(int),cudaMemcpyDeviceToDevice);
  cudaFree(nlist_tmp);

  // cudaMemcpy(*D4D,*H4D,sizeof(struct_nblist),cudaMemcpyHostToDevice);
}


void free_nblist(struct_nblist* H4H,struct_nblist* H4D)
{
  free(H4H->bin);
  free(H4H->iblock);
  // free(H4H->Nblock);
  // free(H4H->block);
  cudaFree(H4D->bin);
  cudaFree(H4D->iblock);
  // cudaFree(H4D->Nblock);
  // cudaFree(H4D->block);

  // free(H4H->itile);
  // free(H4H->jtile);
  free(H4H->ijtile);
  free(H4H->shift);
  // free(H4H->iB);
  // free(H4H->jB);
  // free(H4H->blocal);
  // free(H4H->N_local);
  free(H4H->nlist);
  // cudaFree(H4D->itile);
  // cudaFree(H4D->jtile);
  cudaFree(H4D->ijtile);
  cudaFree(H4D->shift);
  // cudaFree(H4D->iB);
  // cudaFree(H4D->jB);
  // cudaFree(H4D->blocal);
  // cudaFree(H4D->N_local);
  cudaFree(H4D->nlist);

  free(H4H);
  free(H4D);
  // cudaFree(D4D);
}


// Subroutines
int eq_col(int b1[2],int b2[2])
{
  return (b1[0]==b2[0]) && (b1[1]==b2[1]);
}


int ge_col(int b1[2],int b2[2])
{
  int cmp;
  cmp=2*((b1[0]>b2[0])-(b1[0]<b2[0]));
  cmp+=1*((b1[1]>b2[1])-(b1[1]<b2[1]));
  return cmp>=0;
}


void find_col(int col[2],int colind[2],struct_bin *bin,int *iblock,int Nblocks)
{
  int ibelow=-1;
  int iabove=Nblocks;
  int itest;
  int testcol[2];

  // Find lower bound
  while (iabove-ibelow>1) {
    itest=ibelow+(iabove-ibelow)/2;
    testcol[0]=bin[iblock[itest]].bx;
    testcol[1]=bin[iblock[itest]].by;
    if (ge_col(testcol,col)) {
      iabove=itest;
    } else {
      ibelow=itest;
    }
  }
  colind[0]=iabove;
  
  ibelow=colind[0]-1;
  iabove=Nblocks;

  // Find upper bound
  while (iabove-ibelow>1) {
    itest=ibelow+(iabove-ibelow)/2;
    testcol[0]=bin[iblock[itest]].bx;
    testcol[1]=bin[iblock[itest]].by;
    if (ge_col(col,testcol)) {
      ibelow=itest;
    } else {
      iabove=itest;
    }
  }
  colind[1]=iabove;
}


void find_cols(int col[2],int colind[5][2],struct_bin *bin,int *iblock,int Nblocks,vec box,vec shifts[5],int ncol[2])
{
  int i,j;
  int dcol[5][2]={{0,0},{1,1},{1,0},{1,-1},{0,1}};
  int targcol[2];

  for (i=0; i<5; i++) {
    for (j=0; j<2; j++) {
      targcol[j]=col[j]+dcol[i][j];
      if (targcol[j]>=ncol[j]) {
        targcol[j]-=ncol[j];
        shifts[i][j]=-box[j];
      } else if (targcol[j]<0) {
        targcol[j]+=ncol[j];
        shifts[i][j]=box[j];
      } else {
        shifts[i][j]=0;
      }
    }
    shifts[i][2]=0;
    find_col(targcol,colind[i],bin,iblock,Nblocks);
  }
}


// nt=include_col(nt,colind[0],shifts[0],zmid,zmax,box[2],H4H,H4D);
int include_col(int nt,int i,int colind[2],vec shift,real zmin,real zmax,real boxz,struct_nblist *H4H,struct_nblist *H4D)
{
  int j;
  vec shiftbuf;
  real zlimlow;
  real zlimup;

  vec_copy(shiftbuf,shift);

  if (zmin < 0) {
    shiftbuf[2]=boxz;
    for (j=colind[0]; j<colind[1]; j++) {
      zlimlow=H4H->bin[H4H->iblock[j]].z-boxz;
      // zlimup=H4H->bin[H4H->iblock[j]+H4H->Nblock[j]-1].z-boxz;
      zlimup=H4H->bin[H4H->iblock[j+1]-1].z-boxz;
      if (zlimup>=zmin && zlimlow<=zmax) {
        if (nt >= H4H->mNtiles) {
          realloc_nblist(&H4H,&H4D);
        }
        // H4H->itile[nt]=i;
        // H4H->jtile[nt]=j;
        H4H->ijtile[nt].i=i;
        H4H->ijtile[nt].j=j;
        vec_copy(H4H->shift[nt],shiftbuf);
        nt++;
      }
    }
  }
  {
    shiftbuf[2]=0;
    for (j=colind[0]; j<colind[1]; j++) {
      zlimlow=H4H->bin[H4H->iblock[j]].z;
      // zlimup=H4H->bin[H4H->iblock[j]+H4H->Nblock[j]-1].z;
      zlimup=H4H->bin[H4H->iblock[j+1]-1].z;
      if (zlimup>=zmin && zlimlow<=zmax) {
        if (nt >= H4H->mNtiles) {
          realloc_nblist(&H4H,&H4D);
        }
        // H4H->itile[nt]=i;
        // H4H->jtile[nt]=j;
        H4H->ijtile[nt].i=i;
        H4H->ijtile[nt].j=j;
        vec_copy(H4H->shift[nt],shiftbuf);
        nt++;
      }
    }
  }
  if (zmax >= boxz) {
    shiftbuf[2]=-boxz;
    for (j=colind[0]; j<colind[1]; j++) {
      zlimlow=H4H->bin[H4H->iblock[j]].z+boxz;
      // zlimup=H4H->bin[H4H->iblock[j]+H4H->Nblock[j]-1].z+boxz;
      zlimup=H4H->bin[H4H->iblock[j+1]-1].z+boxz;
      if (zlimup>=zmin && zlimlow<=zmax) {
        if (nt >= H4H->mNtiles) {
          realloc_nblist(&H4H,&H4D);
        }
        // H4H->itile[nt]=i;
        // H4H->jtile[nt]=j;
        H4H->ijtile[nt].i=i;
        H4H->ijtile[nt].j=j;
        vec_copy(H4H->shift[nt],shiftbuf);
        nt++;
      }
    }
  }
  return nt;
}

/*
__host__ __device__ static inline
int checkexcl(struct_exclparms *excl,int i,int j)
{
  int a;
  if (excl != NULL) {
    for (a=0; a<excl[i].N[2]; a++) {
      if (j==excl[i].j[a]) {
        return 0;
      }
    }
  }
  return 1;
}
*/

__host__ __device__ static inline
int checkexclhash(struct_exclhash exclhash,int i, int j)
{
  int a,a0,a1;

  a0=exclhash.ind[i*XHASH+(j%XHASH)];
  a1=exclhash.ind[i*XHASH+(j%XHASH)+1];

  for (a=a0; a<a1; a++) {
    if (j==exclhash.data[a]) {
      return 0;
    }
  }
  return 1;
}


// Neighbor searching
void nblist_receive(struct_atoms at,real* x)
{
  cudaMemcpy(x,at.x,at.N*sizeof(real),cudaMemcpyDeviceToHost);
}


/*void nblist_bin(struct_nblist* H4H,real rc,real* box,real* x)
{
  int i,j;
  int N=H4H->N;
  vec dx;

  for (j=0; j<DIM3; j++) {
    H4H->div.b[j]=(int) (box[j]/rc);
    dx[j]=box[j]/H4H->div.b[j];
  }

  for (i=0; i<N; i++) {
    H4H->bin[i].i=i;
    for (j=0; j<DIM3; j++) {
      H4H->bin[i].b[j]=(int) (x[DIM3*i+j]/dx[j]);
    }
  }

  quicksort_bins(H4H->bin,0,N);
}*/


__global__
void nblist_bin_d(struct_nblist H4D,real rc,real* box,real* x)
{
  int i=blockIdx.x*BNB+threadIdx.x;
  int j;
  real shiftbuf;
  vec dx;
  int Nbin[DIM3];

  for (j=0; j<DIM3; j++) {
    Nbin[j]=((int) (box[j]/rc));
    dx[j]=box[j]/Nbin[j];
  }

  if (i<H4D.Nbit) {
    if (i<H4D.N) {
      // Boundary conditions
      for (j=0; j<DIM3; j++) {
        shiftbuf=-box[j]*floor(x[DIM3*i+j]/box[j]);
        x[DIM3*i+j]+=shiftbuf;
      }
      // Binning for sorting
      H4D.bin[i].i=i;
      H4D.bin[i].bx=(int) (x[DIM3*i+0]/dx[0]);
      // if (H4D.bin[i].bx < 0) H4D.bin[i].bx=0;
      if (H4D.bin[i].bx >= Nbin[0]) H4D.bin[i].bx=Nbin[0]-1; // deal with rounding errors
      H4D.bin[i].by=(int) (x[DIM3*i+1]/dx[1]);
      // if (H4D.bin[i].by < 0) H4D.bin[i].by=0;
      if (H4D.bin[i].by >= Nbin[1]) H4D.bin[i].by=Nbin[1]-1; // deal with rounding errors
      H4D.bin[i].z=x[DIM3*i+2];
    } else {
      H4D.bin[i].i=-1;
    }
  }
}


void nblist_bin(cudaStream_t strm,struct_nblist* H4D,struct_nblist* H4H,real rc,real* box,real* boxhost,real* x)
{
  nblist_bin_d <<< H4D->Nbit/BNB, BNB, 0, strm >>> (H4D[0],rc,box,x);

  bitonic_sort(strm,H4D);
  
  cudaMemcpyAsync(H4H->bin,H4D->bin,H4D->N*sizeof(struct_bin),cudaMemcpyDeviceToHost,strm);
}



void nblist_block(struct_nblist* H4H,real rc,vec box)
{
  int i,ii,iprev;
  int N=H4H->N;
  int TN=H4H->TN;
#define BLOCK_PAD_FACTOR 0.9999
  real maxz=box[2]*(BLOCK_PAD_FACTOR/2)-rc;
  struct_bin *b=H4H->bin;
  i=0;
  ii=0;
  H4H->iblock[ii]=i;
  iprev=i;
  for (i=1; i<N; i++) {
    if (i>=iprev+TN || b[i].bx!=b[iprev].bx || b[i].by!=b[iprev].by || b[i].z-b[iprev].z>=maxz) {
      // H4H->Nblock[ii]=(i-iprev);
      ii++;
      H4H->iblock[ii]=i;
      iprev=i;
    }
  }
  // H4H->Nblock[ii]=(i-iprev);
  ii++;
  H4H->iblock[ii]=i;
  H4H->Nblocks=ii;
}


void nblist_tile(struct_nblist* H4H,struct_nblist* H4D,real rc,real* box)
{
  // int i,im,j1,j2,nt;
  int i,im,nt;
  // struct_bin b1;
  // struct_bin b2;
  // vec shift;
  int prevcol[2]={-1,-1};
  int col[2]; // column of grid (bx and by)
  int ncol[2];
  int colind[5][2];
  int selfcolind[2];
  vec shifts[5];
  // real zmin, zmid, zmax;
  real zmin, zmax;

  nt=0;

  ncol[0]=(int) (box[0]/rc);
  ncol[1]=(int) (box[1]/rc);

  for (i=0; i<H4H->Nblocks; i++) {
    col[0]=H4H->bin[H4H->iblock[i]].bx;
    col[1]=H4H->bin[H4H->iblock[i]].by;
    zmin=H4H->bin[H4H->iblock[i]].z-rc;
    // zmax=H4H->bin[H4H->iblock[i]+H4H->Nblock[i]-1].z+rc;
    zmax=H4H->bin[H4H->iblock[i+1]-1].z+rc;
    // zmid=(zmax+zmin)/2;
    // #warning "Method for identifying neighbors in self column isn't very good (right now it's too generous, before it sometimes missed self interactions for 1 particles tiles which are crucial for electrostatics.)"
    // zmid=H4H->bin[H4H->iblock[i]].z;
    // void find_cols(int col[2],int colind[5][2],struct_bin *bin,int *iblock,int Nblocks,vec box,vec shifts[5],int ncol[2])
    if (!eq_col(prevcol,col)) {
      prevcol[0]=col[0];
      prevcol[1]=col[1];
      find_cols(col,colind,H4H->bin,H4H->iblock,H4H->Nblocks,box,shifts,ncol);
    }
    // nt=include_col(nt,i,colind[0],shifts[0],zmid,zmax,box[2],H4H,H4D);
    selfcolind[0]=i;
    selfcolind[1]=colind[0][1];
    nt=include_col(nt,i,selfcolind,shifts[0],zmin,zmax,box[2],H4H,H4D);

    for (im=1; im<5; im++) {
      nt=include_col(nt,i,colind[im],shifts[im],zmin,zmax,box[2],H4H,H4D);
    }
  }
  // fprintf(stderr,"nt=%d\n",nt);
  H4H->Ntiles=nt;
}

/*
void dump_tiles(struct_nblist *H4H)
{
  int i;
  struct_bin b;
  for (i=0; i<H4H->Nblocks; i++) {
    b=H4H->bin[H4H->iblock[i]];
    fprintf(stderr,"Block %d , i %d , b %d %d %d\n",i,b.i,b.b[0],b.b[1],b.b[2]);
  }
  fprintf(stderr,"End+1 %d , i %d , b %d %d %d\n",i,b.i,b.b[0],b.b[1],b.b[2]);

  for (i=0; i<H4H->Ntiles; i++) {
    fprintf(stderr,"Tile %d , i %d , j %d\n",i,H4H->itile[i],H4H->jtile[i]);
  }
  fprintf(stderr,"End1 %d , i %d , j %d\n",i,H4H->itile[i],H4H->jtile[i]);
}
*/
/*
void dump_tiles(struct_nblist *H4H)
{
  int i;
  struct_bin b,bN;

  for (i=0; i<H4H->Nblocks; i++) {
    b=H4H->bin[H4H->iblock[i]];
    bN=H4H->bin[H4H->iblock[i]+H4H->Nblock[i]-1];
    fprintf(stderr,"Block %d , i %d , b %d %d , z %g to %g\n",i,b.i,b.bx,b.by,b.z,bN.z);
  }
  fprintf(stderr,"End+1 %d , i %d , b %d %d , z %g to %g\n",i,b.i,b.bx,b.by,b.z,bN.z);

  for (i=0; i<H4H->Ntiles; i++) {
    fprintf(stderr,"Tile %d , i %d , j %d\n",i,H4H->itile[i],H4H->jtile[i]);
  }
  fprintf(stderr,"End1 %d , i %d , j %d\n",i,H4H->itile[i],H4H->jtile[i]);
}
*/
/*
void dump_nblist(FILE *fp,struct_nblist *H4H,struct_nblist *H4D,real *xhost,real *xdevice,int N,int step)
{
  int a,i,j;
  int i0,j0,iN,jN;
  int ii,jj;
  int nlist;
  real *shift;
  vec dr;
  real r2;

  cudaMemcpy(xhost,xdevice,N*sizeof(real),cudaMemcpyDeviceToHost);
  cudaMemcpy(H4H->iblock,H4D->iblock,H4D->Nblocks*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(H4H->Nblock,H4D->Nblock,H4D->Nblocks*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(H4H->itile,H4D->itile,H4D->Ntiles*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(H4H->jtile,H4D->jtile,H4D->Ntiles*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(H4H->shift,H4D->shift,H4D->Ntiles*sizeof(vec),cudaMemcpyDeviceToHost);
  cudaMemcpy(H4H->nlist,H4D->nlist,H4D->Ntiles*TNB*sizeof(int),cudaMemcpyDeviceToHost);

  fprintf(fp,"Step %d\n",step);
  for (a=0; a<H4H->Ntiles; a++) {
    i0=H4H->iblock[H4H->itile[a]];
    j0=H4H->iblock[H4H->jtile[a]];
    iN=H4H->Nblock[H4H->itile[a]];
    jN=H4H->Nblock[H4H->jtile[a]];
    shift=H4H->shift[a];
    for (i=0; i<iN; i++) {
      ii=H4H->bin[i+i0].i;
      nlist=H4H->nlist[a*TNB+i];
      for (j=0; j<jN; j++) {
        jj=H4H->bin[j+j0].i;
        if (nlist & (1 << j)) {
          vec_subshift_ns(xhost+DIM3*ii,xhost+DIM3*jj,shift,dr);
          r2=vec_mag2(dr);
          fprintf(fp,"%d %d %g\n",ii,jj,sqrt(r2));
        }
      }
    }
  }
  cudaDeviceSynchronize();
}
*/

// __global__ void nblist_check_d(struct_nblist H4D,struct_exclparms* excl,struct_exclhash exclhash,real rc,struct_atoms at)
__global__ void nblist_check_d(struct_nblist H4D,struct_exclhash exclhash,real rc,real* x)
{
  real3 xi;
  int tileOfBlockIdx=threadIdx.x/H4D.TN;
  int tileIdx=tileOfBlockIdx+blockIdx.x*(BNB/H4D.TN);
  // int threadOfTileIdx=(threadIdx.x % H4D.TN);
  int threadOfTileIdx=(threadIdx.x & (H4D.TN-1));
  __shared__ real3 xj[BNB];
  unsigned int nlisti=0;

  struct_ij ij;
  int i,j,ii,jj;
  int iB,i0,iN,jB,j0,jN;
  // __shared__ int i0share[2*BNB/TNB];
  // __shared__ int firstshare[2*BNB/TNB];
  // __shared__ int rankshare[2*BNB/TNB];
  __shared__ int jjshare[BNB];
  vec dr;
  real r2;
  real rc2=rc*rc;

  if (tileIdx<H4D.Ntiles) {
    ij=H4D.ijtile[tileIdx];

    // iB=H4D.itile[tileIdx];
    iB=ij.i;
    i0=H4D.iblock[iB];
    // iN=H4D.Nblock[iB];
    iN=H4D.iblock[iB+1]-i0;
    if (threadOfTileIdx<iN) {
      ii=H4D.bin[i0+threadOfTileIdx].i;
      xi=((real3*) x)[ii];
    } else {
      ii=-1;
    }

    // jB=H4D.jtile[tileIdx];
    jB=ij.j;
    j0=H4D.iblock[jB];
    jN=H4D.iblock[jB+1]-j0;
    if (i0==j0) { // blocks i and j in this tile are identical
      if (threadOfTileIdx<jN) {
        jj=ii;
        // Actually use the full threadIdx; all BNB/TNB tiles are sharing
        xj[threadIdx.x]=xi;
      } else {
        jj=-1;
      }
    } else {
      if (threadOfTileIdx<jN) {
        jj=H4D.bin[j0+threadOfTileIdx].i;
        // Actually use the full threadIdx; all BNB/TNB tiles are sharing
        xj[threadIdx.x]=((real3*) x)[jj];
      } else {
        jj=-1;
      }
    }
    jjshare[threadIdx.x]=jj;

    vec_inc((real*) (&xi),H4D.shift[tileIdx]); // allows use of vec_subtract instead of vec_subshift_ns
  }

  /*if (threadOfTileIdx==0) {
    if (tileIdx<H4D.Ntiles) {
      i0share[tileOfBlockIdx]=i0;
      i0share[BNB/TNB+tileOfBlockIdx]=j0;
    } else {
      i0share[tileOfBlockIdx]=-1;
      i0share[BNB/TNB+tileOfBlockIdx]=-1;
    }
  }*/

  __syncthreads();

  // assert(nlisti==0);

  if (tileIdx<H4D.Ntiles) {

    // if (threadOfTileIdx<iN)
    if (ii>=0) {
      for (i=0; i<H4D.TN; i++) {
        j=(threadOfTileIdx+i) & (H4D.TN-1);
        if (j<jN) {
          jj=jjshare[tileOfBlockIdx*H4D.TN+j];
          // vec_subshift_ns((real*) (&xi),(real*) (&xj[tileOfBlockIdx*TNB+j]),shift,dr);
          vec_subtract((real*) (&xi),(real*) (&xj[tileOfBlockIdx*H4D.TN+j]),dr);
          r2=vec_mag2(dr);
          if (r2<=rc2) {
            // assert(checkexcl(excl,ii,jj)==checkexclhash(exclhash,ii,jj));
            // if (checkexcl(excl,ii,jj))
            if (checkexclhash(exclhash,ii,jj)) {
              if (i0!=j0 || threadOfTileIdx>j) { // Only do half of diagonal tiles, i=threadOfTileIdx
                nlisti+=(1<<j);
              }
            }
          }
        }
      }
      // assert((i0!=j0) || (jN>1) || (nlisti==0));
    }
  }

  // __syncthreads();

/*
  if (tileIdx<H4D.Ntiles && threadOfTileIdx==0) {
    i=tileOfBlockIdx;
    firstshare[i]=0;
    for (j=0; j<i; j++) {
      if (i0share[j]==i0share[i]) {
        firstshare[i]=-1;
      }
    }
    i=BNB/TNB+tileOfBlockIdx;
    firstshare[i]=0;
    for (j=0; j<i; j++) {
      if (i0share[j]==i0share[i]) {
        firstshare[i]=-1;
      }
    }
  } else if (threadOfTileIdx==0) {
    firstshare[tileOfBlockIdx]=-1;
    firstshare[BNB/TNB+tileOfBlockIdx]=-1;
  }

  __syncthreads();

  if (tileIdx<H4D.Ntiles && threadOfTileIdx==0) {
    i=tileOfBlockIdx;
    rankshare[i]=0;
    for (j=0; j<i; j++) {
      if (firstshare[j]==0) {
        rankshare[i]++;
      }
    }
    i=BNB/TNB+tileOfBlockIdx;
    rankshare[i]=0;
    for (j=0; j<i; j++) {
      if (firstshare[j]==0) {
        rankshare[i]++;
      }
    } 
  } else if (threadOfTileIdx==0) {
    rankshare[tileOfBlockIdx]=-1;
  }

  if(threadIdx.x==0) {
    i=0;
    for (j=0; j<(2*BNB/TNB); j++) {
      if (firstshare[j]==0) {
        i++;
      }
    }
    H4D.N_local[blockIdx.x]=i;
  }

  __syncthreads();

  if (tileIdx<H4D.Ntiles && threadOfTileIdx==0) {
    i=tileOfBlockIdx;
    if (firstshare[i]==-1) {
      for (j=i-1; j>=0; j--) {
        if (i0share[j]==i0) {
          rankshare[i]=rankshare[j];
        }
      }
    }
    i=BNB/TNB+tileOfBlockIdx;
    if (firstshare[i]==-1) {
      for (j=i-1; j>=0; j--) {
        if (i0share[j]==j0) {
          rankshare[i]=rankshare[j];
        }
      }
    }
  }
  
  __syncthreads();*/

  if (tileIdx<H4D.Ntiles) {
    /*if (threadOfTileIdx==0) {
      H4D.iB[tileIdx]=rankshare[tileOfBlockIdx];
      H4D.jB[tileIdx]=rankshare[BNB/TNB+tileOfBlockIdx];
      if (firstshare[tileOfBlockIdx]==0) {
        H4D.blocal[2*blockIdx.x*(BNB/TNB)+rankshare[tileOfBlockIdx]]=iB;
      }
      if (firstshare[BNB/TNB+tileOfBlockIdx]==0) {
        H4D.blocal[2*blockIdx.x*(BNB/TNB)+rankshare[BNB/TNB+tileOfBlockIdx]]=jB;
      }
    }*/
    // if (threadOfTileIdx < iN)
    if (ii>=0) {
      // H4D.nlist[tileIdx*H4D.TN+threadOfTileIdx]=nlisti;
      H4D.nlist[blockIdx.x*BNB+threadIdx.x]=nlisti;
    }
  }

  // // if (tileIdx==1 && threadOfTileIdx==0)
  // if (blockIdx.x==0 && threadIdx.x==32) {
  //   H4D.debug[H4D.debug[0]]=nlisti;
  //   H4D.debug[0]++;
  //   H4D.debug[H4D.debug[0]]=H4D.TN;
  //   H4D.debug[0]++;
  //   H4D.debug[H4D.debug[0]]=tileIdx;
  //   H4D.debug[0]++;
  //   H4D.debug[H4D.debug[0]]=threadOfTileIdx;
  //   H4D.debug[0]++;
  // }
}

/*
void nblist_check(struct_nblist* H4H,struct_exclparms* excl,real rc,real* x)
{
  int nt;
  int i,it,i1,iN,ii;
  int j,jt,j1,jN,jj;
  vec shift;
  real rc2=rc*rc;

  for (nt=0; nt<H4H->Ntiles; nt++) {
    vec_copy(shift,H4H->shift[nt]);
    it=H4H->itile[nt];
    jt=H4H->jtile[nt];
    i1=H4H->iblock[it];
    j1=H4H->iblock[jt];
    iN=H4H->Nblock[it];
    jN=H4H->Nblock[jt];
    for (i=0; i<iN; i++) {
      ii=H4H->bin[i+i1].i;
      H4H->nlist[nt*TNB+i]=0;
      for (j=0; j<jN; j++) {
        jj=H4H->bin[j+j1].i;
        if (vec_shiftdist2(x+DIM3*ii,x+DIM3*jj,shift)<=rc2) {
          if (checkexcl(excl,ii,jj)) {
            if (i1!=j1 || i>j) { // Only do half of diagonal tiles
              H4H->nlist[nt*TNB+i]+=(1 << j);
            }
          }
        }
      }
    }
  }
}
*/
/*
void nblist_send(struct_nblist* H4H,struct_nblist* H4D)
{
  H4D->N=H4H->N;
  cudaMemcpy(H4D->bin,H4H->bin,H4H->N*sizeof(struct_bin),cudaMemcpyHostToDevice);

  H4D->Nblocks=H4H->Nblocks;
  cudaMemcpy(H4D->iblock,H4H->iblock,H4H->Nblocks*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(H4D->Nblock,H4H->Nblock,H4H->Nblocks*sizeof(int),cudaMemcpyHostToDevice);

  H4D->Ntiles=H4H->Ntiles;
  cudaMemcpy(H4D->itile,H4H->itile,H4H->Ntiles*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(H4D->jtile,H4H->jtile,H4H->Ntiles*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(H4D->shift,H4H->shift,H4H->Ntiles*sizeof(vec),cudaMemcpyHostToDevice);
  cudaMemcpy(H4D->nlist,H4H->nlist,H4H->Ntiles*TNB*sizeof(int),cudaMemcpyHostToDevice);
}
*/

void nblist_send1(struct_nblist* H4H,struct_nblist* H4D,cudaStream_t strm)
{
  // H4D->N=H4H->N;
  // cudaMemcpy(H4D->bin,H4H->bin,H4H->N*sizeof(struct_bin),cudaMemcpyHostToDevice);

  H4D->Nblocks=H4H->Nblocks;
  // cudaMemcpy(H4D->iblock,H4H->iblock,H4H->Nblocks*sizeof(int),cudaMemcpyHostToDevice);
  // cudaMemcpy(H4D->Nblock,H4H->Nblock,H4H->Nblocks*sizeof(int),cudaMemcpyHostToDevice);
  // cudaMemcpyAsync(H4D->iblock,H4H->iblock,H4H->Nblocks*sizeof(int),cudaMemcpyHostToDevice,strm);
  // cudaMemcpyAsync(H4D->Nblock,H4H->Nblock,H4H->Nblocks*sizeof(int),cudaMemcpyHostToDevice,strm);
  cudaMemcpyAsync(H4D->iblock,H4H->iblock,(H4H->Nblocks+1)*sizeof(int),cudaMemcpyHostToDevice,strm);

  H4D->Ntiles=H4H->Ntiles;
  // cudaMemcpy(H4D->itile,H4H->itile,H4H->Ntiles*sizeof(int),cudaMemcpyHostToDevice);
  // cudaMemcpy(H4D->jtile,H4H->jtile,H4H->Ntiles*sizeof(int),cudaMemcpyHostToDevice);
  // cudaMemcpy(H4D->shift,H4H->shift,H4H->Ntiles*sizeof(vec),cudaMemcpyHostToDevice);
  // cudaMemcpyAsync(H4D->itile,H4H->itile,H4H->Ntiles*sizeof(int),cudaMemcpyHostToDevice,strm);
  // cudaMemcpyAsync(H4D->jtile,H4H->jtile,H4H->Ntiles*sizeof(int),cudaMemcpyHostToDevice,strm);
  cudaMemcpyAsync(H4D->ijtile,H4H->ijtile,H4H->Ntiles*sizeof(struct_ij),cudaMemcpyHostToDevice,strm);
  cudaMemcpyAsync(H4D->shift,H4H->shift,H4H->Ntiles*sizeof(vec),cudaMemcpyHostToDevice,strm);
  // fprintf(stderr,"H4H->Ntiles = %d, (%d blocks)\n",H4H->Ntiles,(H4H->Ntiles+(BNB/TNB)-1)/(BNB/TNB));
}


void neighborsearch(struct_md* md)
{
  struct_atoms at=md->state->atoms[0];
  // real *x=md->state->xhost;
  real rc=md->parms->rcother;
  vec box;
  struct_nblist *H4H=md->state->nblist_H4H;
  struct_nblist *H4D=md->state->nblist_H4D;
  // struct_exclparms *excl=md->parms->exclparms;
  // struct_exclparms *excldev=md->parms->exclparmsdev;
  vec_copy(box,md->state->boxhost);
  gmx_cycles_t start;
  cudaStream_t strm=md->state->stream_default;
  
  start=gmx_cycles_read();
  nblist_bin(strm,H4D,H4H,rc,md->state->box,box,at.x);
  md->times->nblist_bin+=gmx_cycles_read()-start;

  start=gmx_cycles_read();
  // CPU
  nblist_block(H4H,rc,box);
  md->times->nblist_block+=gmx_cycles_read()-start;

  start=gmx_cycles_read();
  // CPU
  nblist_tile(H4H,H4D,rc,box);
  md->times->nblist_tile+=gmx_cycles_read()-start;

  // dump_tiles(H4H);

  start=gmx_cycles_read();
  // CPU
  nblist_send1(H4H,H4D,strm);
  md->times->nblist_send+=gmx_cycles_read()-start;

  start=gmx_cycles_read();
  nblist_check_d <<< (H4H->Ntiles+BNB/H4H->TN-1)/(BNB/H4H->TN), BNB, 0, strm >>> (H4D[0],md->parms->exclhashdev[0],rc,at.x);
  md->times->nblist_check+=gmx_cycles_read()-start;

  // dump_nblist(H4H,H4D,md->state->xhost,at.x,at.N);
}
