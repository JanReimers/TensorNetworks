#ifndef LAPACKSVD_H
#define LAPACKSVD_H

#include <tuple>

template <class T> class Matrix;
template <class T> class DiagonalMatrix;
//
//  SVD decomposition for real*8  matrix.
//
//template <class TM> void LaSVDecomp(TM& A, Vector<double>& s, TM& V);
//std::tuple<Matrix<double>,DiagonalMatrix<double>,Matrix<double> > LaSVDecomp(const Matrix<double>& A);

template <class T> class LapackSVDSolver
{
    typedef        Matrix<T>              MatrixT;
    typedef DiagonalMatrix<double> DiagonalMatrixT;
    typedef std::tuple<MatrixT,DiagonalMatrixT,MatrixT> UsVType;
public:
    LapackSVDSolver() {};
    ~LapackSVDSolver() {};

    UsVType Solve(const MatrixT&, int NumSingularValues,double eps);

};
#endif // LAPACKSVD_H
