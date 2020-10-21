#ifndef ARPACKEIGENSOLVER_H
#define ARPACKEIGENSOLVER_H

#include <tuple>
#include <complex>

template <class T> class Matrix;
template <class T> class SparseMatrix;
template <class T> class Vector;

template <class T> class ArpackEigenSolver
{
    typedef std::complex<double> dcmplx;
    typedef std::tuple<Matrix<dcmplx>,Vector<dcmplx> > UdType;
public:
    ArpackEigenSolver() {};
    ~ArpackEigenSolver() {};

    //Return value is: (EigenValues,EigenVectors)
    UdType SolveNonSym(const Matrix     <T>&, int NumEigenValues,double eps);
    UdType SolveNonSym(const SparseMatrix<T>&, int NumEigenValues,double eps);
private:
    // Generic version
    template <template <typename> class Mat> UdType SolveG(const Mat<T>&, int NumEigenValues,double eps);
};

#endif // ARPACKEIGENSOLVER_H
