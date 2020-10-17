#ifndef LAPACKEIGENSOLVER_H
#define LAPACKEIGENSOLVER_H

#include <tuple>

template <class T> class DMatrix;
template <class T> class Vector;
namespace std {
template <class T> class complex;
}

template <class T> class LapackEigenSolver
{
    typedef std::complex<double> dcmplx;
    typedef Vector <double> VectorR;
    typedef Vector <dcmplx> VectorC;
    typedef DMatrix<T>      MatrixT;
    typedef DMatrix<dcmplx> MatrixC;
    typedef std::tuple<MatrixT,VectorR> UdType;
    typedef std::tuple<MatrixC,VectorC> UdTypeN;
public:
    LapackEigenSolver() {};
    ~LapackEigenSolver() {};

    UdType  Solve      (const MatrixT&, int NumEigenValues,double eps);
    UdTypeN SolveNonSym(const MatrixT&, int NumEigenValues,double eps);
};
#endif // LAPACKEIGENSOLVER_H
