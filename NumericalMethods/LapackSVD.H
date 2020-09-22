#ifndef LAPACKSVD_H
#define LAPACKSVD_H

#include <tuple>

template <class T> class Vector;
template <class T> class DMatrix;
template <class T> class DiagonalMatrix;
//
//  SVD decomposition for real*8  matrix.
//
template <class TM> void LaSVDecomp(TM& A, Vector<double>& s, TM& V);
std::tuple<DMatrix<double>,DiagonalMatrix<double>,DMatrix<double> > LaSVDecomp(const DMatrix<double>& A);

#endif // LAPACKSVD_H
