#include "NumericalMethods/SparseSVDSolver.H"
#include "NumericalMethods/SparseEigenSolver.H"
#include "NumericalMethods/EigenSolver.H"
#include "Containers/SparseMatrix.H"
#include "oml/matrix.h"
#include "oml/vector.h"
#include "oml/fakedouble.h"

template <class T> typename EigenSolver<T>::UdTypeN
EigenSolver<T>::SolveLeft_NonSym(const MatrixT& A,double eps, int NumEigenValues)
{
    MatrixT Adagger=Transpose(conj(A));
    auto [U,d]=SolveRightNonSym(Adagger,eps,NumEigenValues);
    return std::make_tuple(conj(U),conj(d));
}

template <class T> typename SparseEigenSolver<T>::UdTypeN
SparseEigenSolver<T>::SolveLeft_NonSym(const SparseMatrixT& A,double eps, int NumEigenValues)
{
    SparseMatrixT Adagger=~A;
    auto [U,d]=SolveRightNonSym(Adagger,eps,NumEigenValues);
    return std::make_tuple(conj(U),conj(d));
}

//
//  static variable. Kludge for getting the matrix into the Mat*Vec routines.
//
template<class T> const SparseMatrix<T>* SparseEigenSolver<T>::theSparseMatrix = 0;
template<class T> const       Matrix<T>* SparseEigenSolver<T>::theDenseMatrix = 0;
template<class T> const SparseMatrix<T>* SparseSVDSolver<T>::theSparseMatrix = 0;
template<class T> const       Matrix<T>* SparseSVDSolver<T>::theDenseMatrix = 0;
//
//  Used to pass client pointers in mat*vec functions
//
template <class T> const SparseEigenSolverClient<T>* SparseEigenSolverClient<T>::theClient;
template <class T> const SparseSVDSolverClient<T>* SparseSVDSolverClient<T>::theClient;

typedef std::complex<double> dcmplx;

template class SparseSVDSolver<double>;
template class SparseSVDSolver<dcmplx>;
template class EigenSolver<double>;
template class EigenSolver<dcmplx>;
template class SparseEigenSolver<double>;
template class SparseEigenSolver<dcmplx>;
template class SparseSVDSolverClient<double>;
template class SparseSVDSolverClient<dcmplx>;
template class SparseEigenSolverClient<double>;
template class SparseEigenSolverClient<dcmplx>;

