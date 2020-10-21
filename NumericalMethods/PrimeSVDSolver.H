#ifndef PRIMESVDSOLVER_H
#define PRIMESVDSOLVER_H

#include "oml/vector.h"
#include "oml/diagonalmatrix.h"
#include <tuple>

struct primme_svds_params;
template <class T> class SparseMatrix;
template <class T> class Matrix;
//
//  Client code can derive from this and control the matvec multiplication operations
//
template <class T> class PrimeSVDSolverClient
{
public:
    virtual int  GetNumRows() const=0;
    virtual int  GetNumCols() const=0;
    virtual void DoMatVecContraction (int N, T* xvec, T* yvec,int transpose) const=0;

    static const PrimeSVDSolverClient* theClient;
};
//
//  Factor dense or sparse matrices using the PRIMME library
//  For efficiency client code should keep an instance of PrimeSVDSolver alive
//  between iterations so that initial guesses get used.
//
template <class T> class PrimeSVDSolver
{
    typedef         Vector<T>              VectorT;
    typedef        Matrix<T>              MatrixT;
    typedef   SparseMatrix<T>        SparseMatrixT;
    typedef DiagonalMatrix<double> DiagonalMatrixT;
    typedef std::tuple<MatrixT,DiagonalMatrixT,MatrixT> UsVType;
public:
    PrimeSVDSolver();
    ~PrimeSVDSolver();

    // Get lowest N eigen values and vectors with no initial guess
    UsVType Solve(const MatrixT&                , int NumSingularValues,double eps);
    UsVType Solve(const SparseMatrixT&          , int NumSingularValues,double eps);
    UsVType Solve(const PrimeSVDSolverClient<T>*, int NumSingularValues,double eps);

    static const SparseMatrixT* theSparseMatrix;
    static const       MatrixT* theDenseMatrix;
private:
    UsVType Solve(primme_svds_params&);


    DiagonalMatrixT  itsSingularValues;
    VectorT          itsSingularVectors; //Packed storage of U and VT (or Vdagger)
    int              itsNumGuesses;
};

#endif // PRIMESVDSOLVER_H
