#include "sort.h"

#include "nblist.h"
#include "error.h"

#include <stdio.h>

/*
__global__ void keybonds(int N,struct_bondparms *bonds,int (*keys)[2])
{
  i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i<N) {
    keys[i][0]=bonds.i;
    keys[i][1]=bonds.j;
  }
}


__global__ void countbins(int N,int ind,int s,int (*keys)[2],int (*hist)[BSORT])
{
  __shared__ lhist[BSORT];
  int mask=((BSORT-1)<<s);
  
  lhist[threadIdx.x]=0;

  for (i=0; i<CSORT; i++) {
    ii=(blockIdx.x*blockDim.x+threadIdx.x)*CSORT+i;
    if (ii < N) {
      bin=((keys[ii][ind] & mask) >> s);
      atomicAdd(&(lhist[bin]),1);
    }
  }

  __syncthreads();

  hist[blockIdx.x][threadIdx.x]=lhist[threadIdx.x];
}


// WARNING - suboptimal - an ugly tree would be faster if N starts getting big. For example with 1329 atoms it's 2 though.
__global__ void cumbins(int N
{
  
}


struct_sort* alloc_bondsort(int N,struct_bondparms *bonds)
{
  struct_sort *bondsort;

  bondsort=(struct_bondparmsort*) malloc(sizeof(struct_bondparmsort));
  bondsort->N=N;
  bondsort->n_keys=2;
  bondsort->n_data=sizeof(struct_bondparms);
  cudaMalloc(&(bondsort->keys),N*sizeof(int[2]));
  cudaMalloc(&(bondsort->keys_buf),N*sizeof(int[2]));
  bondsort->bonds=(void*) bonds;
  cudaMalloc(&(bondsort->data_buf),N*sizeof(struct_bondparms));
  cudaMalloc(&(bondsort->hist),(N+SORT_THREADS_PER_BLOCK*SORT_ATOMS_PER_THREAD-1)/(SORT_THREADS_PER_BLOCK*SORT_ATOMS_PER_THREAD)*sizeof(int[SORT_THREADS_PER_BLOCK]));
}


void free_bondsort(struct_sort *bondsort)
{
  cudaFree(bondsort->keys);
  cudaFree(bondsort->keys_buf);
  cudaFree(bondsort->data_buf);
  cudaFree(bondsort->hist);
  free(bondsort);
}


void do_bondsort(int N,int Natom,struct_bondparmsort *bondsort)
{
  // sort on j, ind=1  - least significant - first, then on i ind=0
  for (ind=1; ind>=0; ind--) {
    mask=BSORT-1;
    for (s=0; (mask<<s)&Natom; s+=5) { //CONST - ln2(BSORT)
      keybonds <<< (N+CSORT*BSORT-1)/(CSORT*BSORT), BSORT >>> (N,bondsort->bonds,bondsort->keys);
      countbins
      
    }
  }
}*/


void swap_ints(int *ints,int i1,int i2)
{
  int buffer;
  buffer=ints[i1];
  ints[i1]=ints[i2];
  ints[i2]=buffer;
}


int compare_ints(int *ints,int i1,int i2)
{
  int b1=ints[i1];
  int b2=ints[i2];
  if (b1<b2) {
    return 0;
  } else if (b1>b2) {
    return 1;
  }
  // If all else fails
  return 0;
}


int sift_ints(int *ints,int i1,int i2)
{
  int imid=i1+(i2-1-i1)/2;
  int q1,q2;
  swap_ints(ints,imid,i2-1); // place middle at end (to make already sorted lists faster)
  imid=i2-1;
  i2--;
  q1=compare_ints(ints,i1,imid);
  q2=compare_ints(ints,imid,i2-1);
  while (i1<i2-1) {
    if (q1==0) {
      i1++;
      q1=compare_ints(ints,i1,imid);
    } else if (q2==0) {
      i2--;
      q2=compare_ints(ints,imid,i2-1);
    } else {
      swap_ints(ints,i1,i2-1);
      q1=0;
      q2=0;
    }
  }
  if (q1==0) {
    swap_ints(ints,i1+1,imid);
    return i1+1;
  } else {
    swap_ints(ints,i1,imid);
    return i1;
  }
}


void quicksort_ints(int *ints,int i1,int i2)
{
  if (i1<i2-1) {
    int imid=sift_ints(ints,i1,i2);
    quicksort_ints(ints,i1,imid);
    quicksort_ints(ints,imid+1,i2);
  }
}


int unique_ints(int *ints,int N)
{
  int i1, i2;
  int prev;
  prev=ints[0];
  i2=1;
  for (i1=1; i1<N; i1++) {
    if (ints[i1] != prev) {
      prev=ints[i1];
      ints[i2]=ints[i1];
      i2++;
    }
  }
  return i2;
}


int getlocalindex(int *ints,int N,int ind)
{
  int i;
  char message[MAXLENGTH];
  for (i=0; i<N; i++) {
    if (ints[i]==ind) {
      return i;
    }
  }
  sprintf(message,"Something went wrong while processing bonded/angle/dihedral interactions for use in GPU blocks. Most likely cause is heap corruption. Global index %d could not be found among local indices.\n",ind+1);
  fatalerror(message);
  return i; // This should never be used, it's here to prevent compiler warnings
}


void swap_bonds(struct_bondparms *bonds,int i1,int i2)
{
  struct_bondparms buffer;
  buffer=bonds[i1];
  bonds[i1]=bonds[i2];
  bonds[i2]=buffer;
}


int compare_bonds(struct_bondparms *bonds,int i1,int i2)
{
  struct_bondparms b1=bonds[i1];
  struct_bondparms b2=bonds[i2];
  int min1=(b1.i<b1.j) ? b1.i : b1.j;
  int min2=(b2.i<b2.j) ? b2.i : b2.j;
  int max1,max2;
  if (min1<min2) {
    return 0;
  } else if (min1>min2) {
    return 1;
  }
  max1=(b1.i>b1.j) ? b1.i : b1.j;
  max2=(b2.i>b2.j) ? b2.i : b2.j;
  if (max1<max2) {
    return 0;
  } else if (max1>max2) {
    return 1;
  }
  // If all else fails
  return 0;
}


int sift_bonds(struct_bondparms *bonds,int i1,int i2)
{
  int imid=i1+(i2-1-i1)/2;
  int q1,q2;
  swap_bonds(bonds,imid,i2-1); // place middle at end (to make already sorted lists faster)
  imid=i2-1;
  i2--;
  q1=compare_bonds(bonds,i1,imid);
  q2=compare_bonds(bonds,imid,i2-1);
  while (i1<i2-1) {
    if (q1==0) {
      i1++;
      q1=compare_bonds(bonds,i1,imid);
    } else if (q2==0) {
      i2--;
      q2=compare_bonds(bonds,imid,i2-1);
    } else {
      swap_bonds(bonds,i1,i2-1);
      q1=0;
      q2=0;
    }
  }
  if (q1==0) {
    swap_bonds(bonds,i1+1,imid);
    return i1+1;
  } else {
    swap_bonds(bonds,i1,imid);
    return i1;
  }
}


void quicksort_bonds(struct_bondparms *bonds,int i1,int i2)
{
  if (i1<i2-1) {
    int imid=sift_bonds(bonds,i1,i2);
    quicksort_bonds(bonds,i1,imid);
    quicksort_bonds(bonds,imid+1,i2);
  }
}


void swap_angles(struct_angleparms *angles,int i1,int i2)
{
  struct_angleparms buffer;
  buffer=angles[i1];
  angles[i1]=angles[i2];
  angles[i2]=buffer;
}


int compare_angles(struct_angleparms *angles,int i1,int i2)
{
  struct_angleparms b1=angles[i1];
  struct_angleparms b2=angles[i2];
  int min1,min2,max1,max2,mid1,mid2;
  min1=(b1.i<b1.j) ? b1.i : b1.j;
  min1=(min1<b1.k) ? min1 : b1.k;
  min2=(b2.i<b2.j) ? b2.i : b2.j;
  min2=(min2<b2.k) ? min2 : b2.k;
  if (min1<min2) {
    return 0;
  } else if (min1>min2) {
    return 1;
  }
  max1=(b1.i>b1.j) ? b1.i : b1.j;
  max1=(max1>b1.k) ? max1 : b1.k;
  max2=(b2.i>b2.j) ? b2.i : b2.j;
  max2=(max2>b2.k) ? max2 : b2.k;
  mid1=max1;
  mid1=(b1.i>min1 && b1.i<mid1) ? b1.i : mid1;
  mid1=(b1.j>min1 && b1.j<mid1) ? b1.j : mid1;
  mid1=(b1.k>min1 && b1.k<mid1) ? b1.k : mid1;
  mid2=max2;
  mid2=(b2.i>min2 && b2.i<mid2) ? b2.i : mid2;
  mid2=(b2.j>min2 && b2.j<mid2) ? b2.j : mid2;
  mid2=(b2.k>min2 && b2.k<mid2) ? b2.k : mid2;
  if (mid1<mid2) {
    return 0;
  } else if (mid1>mid2) {
    return 1;
  }
  if (max1<max2) {
    return 0;
  } else if (max1>max2) {
    return 1;
  }
  // If all else fails
  return 0;
}


int sift_angles(struct_angleparms *angles,int i1,int i2)
{
  int imid=i1+(i2-1-i1)/2;
  int q1,q2;
  swap_angles(angles,imid,i2-1); // place middle at end (to make already sorted lists faster)
  imid=i2-1;
  i2--;
  q1=compare_angles(angles,i1,imid);
  q2=compare_angles(angles,imid,i2-1);
  while (i1<i2-1) {
    if (q1==0) {
      i1++;
      q1=compare_angles(angles,i1,imid);
    } else if (q2==0) {
      i2--;
      q2=compare_angles(angles,imid,i2-1);
    } else {
      swap_angles(angles,i1,i2-1);
      q1=0;
      q2=0;
    }
  }
  if (q1==0) {
    swap_angles(angles,i1+1,imid);
    return i1+1;
  } else {
    swap_angles(angles,i1,imid);
    return i1;
  }
}


void quicksort_angles(struct_angleparms *angles,int i1,int i2)
{
  if (i1<i2-1) {
    int imid=sift_angles(angles,i1,i2);
    quicksort_angles(angles,i1,imid);
    quicksort_angles(angles,imid+1,i2);
  }
}


void swap_dihs(struct_dihparms *dihs,int i1,int i2)
{
  struct_dihparms buffer;
  buffer=dihs[i1];
  dihs[i1]=dihs[i2];
  dihs[i2]=buffer;
}


int compare_dihs(struct_dihparms *dihs,int i1,int i2)
{
  struct_dihparms b1=dihs[i1];
  struct_dihparms b2=dihs[i2];
  int min1,min2,max1,max2,mid1,mid2;
  min1=(b1.i<b1.j) ? b1.i : b1.j;
  min1=(min1<b1.k) ? min1 : b1.k;
  min1=(min1<b1.l) ? min1 : b1.l;
  min2=(b2.i<b2.j) ? b2.i : b2.j;
  min2=(min2<b2.k) ? min2 : b2.k;
  min2=(min2<b2.l) ? min2 : b2.l;
  if (min1<min2) {
    return 0;
  } else if (min1>min2) {
    return 1;
  }
  max1=(b1.i>b1.j) ? b1.i : b1.j;
  max1=(max1>b1.k) ? max1 : b1.k;
  max1=(max1>b1.l) ? max1 : b1.l;
  max2=(b2.i>b2.j) ? b2.i : b2.j;
  max2=(max2>b2.k) ? max2 : b2.k;
  max2=(max2>b2.l) ? max2 : b2.l;
  mid1=max1;
  mid1=(b1.i>min1 && b1.i<mid1) ? b1.i : mid1;
  mid1=(b1.j>min1 && b1.j<mid1) ? b1.j : mid1;
  mid1=(b1.k>min1 && b1.k<mid1) ? b1.k : mid1;
  mid1=(b1.l>min1 && b1.l<mid1) ? b1.l : mid1;
  mid2=max2;
  mid2=(b2.i>min2 && b2.i<mid2) ? b2.i : mid2;
  mid2=(b2.j>min2 && b2.j<mid2) ? b2.j : mid2;
  mid2=(b2.k>min2 && b2.k<mid2) ? b2.k : mid2;
  mid2=(b2.l>min2 && b2.l<mid2) ? b2.l : mid2;
  if (mid1<mid2) {
    return 0;
  } else if (mid1>mid2) {
    return 1;
  }
  min1=mid1;
  mid1=max1;
  mid1=(b1.i>min1 && b1.i<mid1) ? b1.i : mid1;
  mid1=(b1.j>min1 && b1.j<mid1) ? b1.j : mid1;
  mid1=(b1.k>min1 && b1.k<mid1) ? b1.k : mid1;
  mid1=(b1.l>min1 && b1.l<mid1) ? b1.l : mid1;
  min2=mid2;
  mid2=max2;
  mid2=(b2.i>min2 && b2.i<mid2) ? b2.i : mid2;
  mid2=(b2.j>min2 && b2.j<mid2) ? b2.j : mid2;
  mid2=(b2.k>min2 && b2.k<mid2) ? b2.k : mid2;
  mid2=(b2.l>min2 && b2.l<mid2) ? b2.l : mid2;
  if (mid1<mid2) {
    return 0;
  } else if (mid1>mid2) {
    return 1;
  }
  if (max1<max2) {
    return 0;
  } else if (max1>max2) {
    return 1;
  }
  // If all else fails
  return 0;
}


int sift_dihs(struct_dihparms *dihs,int i1,int i2)
{
  int imid=i1+(i2-1-i1)/2;
  int q1,q2;
  swap_dihs(dihs,imid,i2-1); // place middle at end (to make already sorted lists faster)
  imid=i2-1;
  i2--;
  q1=compare_dihs(dihs,i1,imid);
  q2=compare_dihs(dihs,imid,i2-1);
  while (i1<i2-1) {
    if (q1==0) {
      i1++;
      q1=compare_dihs(dihs,i1,imid);
    } else if (q2==0) {
      i2--;
      q2=compare_dihs(dihs,imid,i2-1);
    } else {
      swap_dihs(dihs,i1,i2-1);
      q1=0;
      q2=0;
    }
  }
  if (q1==0) {
    swap_dihs(dihs,i1+1,imid);
    return i1+1;
  } else {
    swap_dihs(dihs,i1,imid);
    return i1;
  }
}


void quicksort_dihs(struct_dihparms *dihs,int i1,int i2)
{
  if (i1<i2-1) {
    int imid=sift_dihs(dihs,i1,i2);
    quicksort_dihs(dihs,i1,imid);
    quicksort_dihs(dihs,imid+1,i2);
  }
}


struct_bondblock* alloc_bondblock(struct_bondparms *bondparms,int N)
{
  struct_bondblock *bondblock;
  int *local, *N_local;
  struct_bondparms *bonds;
  int i,j,ind;

  // bondparms is on the device, bonds is here on the host
  bonds=(struct_bondparms*) calloc(N,sizeof(struct_bondparms));
  cudaMemcpy(bonds,bondparms,N*sizeof(struct_bondparms),cudaMemcpyDeviceToHost);

  quicksort_bonds(bonds,0,N);

  local=(int*) calloc((N+BB-1)/BB,sizeof(int[2*BB]));
  N_local=(int*) calloc((N+BB-1)/BB,sizeof(int));
  for (i=0; i<(N+BB-1)/BB; i++) {
    N_local[i]=0;
    for (j=0; j<BB; j++) {
      ind=i*BB+j;
      if (ind<N) {
        local[2*ind+0]=bonds[ind].i;
        local[2*ind+1]=bonds[ind].j;
        N_local[i]+=2;
      }
    }
    quicksort_ints(local,2*BB*i,2*BB*i+N_local[i]);
    N_local[i]=unique_ints(local+2*BB*i,N_local[i]);
    for (j=0; j<BB; j++) {
      ind=i*BB+j;
      if (ind<N) {
        bonds[ind].i=getlocalindex(local+2*BB*i,N_local[i],bonds[ind].i);
        bonds[ind].j=getlocalindex(local+2*BB*i,N_local[i],bonds[ind].j);
      }
    }
  }

  cudaMemcpy(bondparms,bonds,N*sizeof(struct_bondparms),cudaMemcpyHostToDevice);
  free(bonds);

  bondblock=(struct_bondblock*) malloc(sizeof(struct_bondblock));

  cudaMalloc(&(bondblock->local),(N+BB-1)/BB*2*BB*sizeof(int));
  cudaMemcpy(bondblock->local,local,(N+BB-1)/BB*2*BB*sizeof(int),cudaMemcpyHostToDevice);
  free(local);
  cudaMalloc(&(bondblock->N_local),(N+BB-1)/BB*sizeof(int));
  cudaMemcpy(bondblock->N_local,N_local,(N+BB-1)/BB*sizeof(int),cudaMemcpyHostToDevice);
  free(N_local);

  return bondblock;
}


void free_bondblock(struct_bondblock* bondblock)
{
  cudaFree(bondblock->local);
  cudaFree(bondblock->N_local);
  free(bondblock);
}


struct_angleblock* alloc_angleblock(struct_angleparms *angleparms,int N)
{
  struct_angleblock *angleblock;
  int *local, *N_local;
  struct_angleparms *angles;
  int i,j,ind;

  // angleparms is on the device, angles is here on the host
  angles=(struct_angleparms*) calloc(N,sizeof(struct_angleparms));
  cudaMemcpy(angles,angleparms,N*sizeof(struct_angleparms),cudaMemcpyDeviceToHost);

  quicksort_angles(angles,0,N);

  local=(int*) calloc((N+BB-1)/BB,sizeof(int[3*BB]));
  N_local=(int*) calloc((N+BB-1)/BB,sizeof(int));
  for (i=0; i<(N+BB-1)/BB; i++) {
    N_local[i]=0;
    for (j=0; j<BB; j++) {
      ind=i*BB+j;
      if (ind<N) {
        local[3*ind+0]=angles[ind].i;
        local[3*ind+1]=angles[ind].j;
        local[3*ind+2]=angles[ind].k;
        N_local[i]+=3;
      }
    }
    quicksort_ints(local,3*BB*i,3*BB*i+N_local[i]);
    N_local[i]=unique_ints(local+3*BB*i,N_local[i]);
    for (j=0; j<BB; j++) {
      ind=i*BB+j;
      if (ind<N) {
        angles[ind].i=getlocalindex(local+3*BB*i,N_local[i],angles[ind].i);
        angles[ind].j=getlocalindex(local+3*BB*i,N_local[i],angles[ind].j);
        angles[ind].k=getlocalindex(local+3*BB*i,N_local[i],angles[ind].k);
      }
    }
  }

  cudaMemcpy(angleparms,angles,N*sizeof(struct_angleparms),cudaMemcpyHostToDevice);
  free(angles);

  angleblock=(struct_angleblock*) malloc(sizeof(struct_angleblock));

  cudaMalloc(&(angleblock->local),(N+BB-1)/BB*3*BB*sizeof(int));
  cudaMemcpy(angleblock->local,local,(N+BB-1)/BB*3*BB*sizeof(int),cudaMemcpyHostToDevice);
  free(local);
  cudaMalloc(&(angleblock->N_local),(N+BB-1)/BB*sizeof(int));
  cudaMemcpy(angleblock->N_local,N_local,(N+BB-1)/BB*sizeof(int),cudaMemcpyHostToDevice);
  free(N_local);

  return angleblock;
}


void free_angleblock(struct_angleblock* angleblock)
{
  cudaFree(angleblock->local);
  cudaFree(angleblock->N_local);
  free(angleblock);
}


struct_dihblock* alloc_dihblock(struct_dihparms *dihparms,int N)
{
  struct_dihblock *dihblock;
  int *local, *N_local;
  struct_dihparms *dihs;
  int i,j,ind;

  // dihparms is on the device, dihs is here on the host
  dihs=(struct_dihparms*) calloc(N,sizeof(struct_dihparms));
  cudaMemcpy(dihs,dihparms,N*sizeof(struct_dihparms),cudaMemcpyDeviceToHost);

  quicksort_dihs(dihs,0,N);

  local=(int*) calloc((N+BB-1)/BB,sizeof(int[4*BB]));
  N_local=(int*) calloc((N+BB-1)/BB,sizeof(int));
  for (i=0; i<(N+BB-1)/BB; i++) {
    N_local[i]=0;
    for (j=0; j<BB; j++) {
      ind=i*BB+j;
      if (ind<N) {
        local[4*ind+0]=dihs[ind].i;
        local[4*ind+1]=dihs[ind].j;
        local[4*ind+2]=dihs[ind].k;
        local[4*ind+3]=dihs[ind].l;
        N_local[i]+=4;
      }
    }
    quicksort_ints(local,4*BB*i,4*BB*i+N_local[i]);
    N_local[i]=unique_ints(local+4*BB*i,N_local[i]);
    for (j=0; j<BB; j++) {
      ind=i*BB+j;
      if (ind<N) {
        dihs[ind].i=getlocalindex(local+4*BB*i,N_local[i],dihs[ind].i);
        dihs[ind].j=getlocalindex(local+4*BB*i,N_local[i],dihs[ind].j);
        dihs[ind].k=getlocalindex(local+4*BB*i,N_local[i],dihs[ind].k);
        dihs[ind].l=getlocalindex(local+4*BB*i,N_local[i],dihs[ind].l);
      }
    }
  }

  cudaMemcpy(dihparms,dihs,N*sizeof(struct_dihparms),cudaMemcpyHostToDevice);
  free(dihs);

  dihblock=(struct_dihblock*) malloc(sizeof(struct_dihblock));

  cudaMalloc(&(dihblock->local),(N+BB-1)/BB*4*BB*sizeof(int));
  cudaMemcpy(dihblock->local,local,(N+BB-1)/BB*4*BB*sizeof(int),cudaMemcpyHostToDevice);
  free(local);
  cudaMalloc(&(dihblock->N_local),(N+BB-1)/BB*sizeof(int));
  cudaMemcpy(dihblock->N_local,N_local,(N+BB-1)/BB*sizeof(int),cudaMemcpyHostToDevice);
  free(N_local);

  return dihblock;
}


void free_dihblock(struct_dihblock* dihblock)
{
  cudaFree(dihblock->local);
  cudaFree(dihblock->N_local);
  free(dihblock);
}
