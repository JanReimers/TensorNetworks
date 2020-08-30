#ifndef LAPACKSVD_H
#define LAPACKSVD_H

template <class T> class Vector;
//
//  SVD decomposition for real*8  matrix.
//
template <class TM> void LaSVDecomp(TM& A, Vector<double>& s, TM& V);

#endif // LAPACKSVD_H
