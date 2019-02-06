#ifndef MD_RNA_IONS_ERROR_H
#define MD_RNA_IONS_ERROR_H

void fatalerror(char *message);

void cudaCheckError(const char fnm[],int line);

#endif
