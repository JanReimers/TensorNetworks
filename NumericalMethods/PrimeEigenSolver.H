#ifndef PRIMEEIGENSOLVER_H
#define PRIMEEIGENSOLVER_H

#include "TensorNetworks/Epsilons.H"
#include "oml/vector.h"
#include "oml/matrix.h"
#include <tuple>

struct primme_params;
template <class T> class SparseMatrix;
template <class T> class Matrix;

//
//  Client code can derive from this and control the matvec multiplication operations
//
template <class T> class PrimeEigenSolverClient
{
public:
    virtual long int GetSize  () const=0; //2^L Hilbert space size
    virtual void     DoMatVecContraction (int N, T* xvec, T* yvec) const=0;

    static const PrimeEigenSolverClient* theClient;
};
//
//  Solve sparse eigen systems using the PRIMME library
//  For efficiency client code should keep and instance of PrimeEigenSolver alive
//  between iterations so that eigen vector initial guess get used.
//
template <class T> class PrimeEigenSolver
{
    typedef    Vector<double>              VectorT;
    typedef        Matrix<T>              MatrixT;
    typedef   SparseMatrix<T>        SparseMatrixT;
    typedef std::tuple<MatrixT,VectorT> UdType;
public:
    PrimeEigenSolver();
    ~PrimeEigenSolver();

    // Get lowest N eigen values and vectors with no initial guess
    UdType Solve(const MatrixT&                  , int NumEigenValues,double eps);
    UdType Solve(const SparseMatrixT&            , int NumEigenValues,double eps);
    UdType Solve(const PrimeEigenSolverClient<T>*, int NumEigenValues,double eps);

    int Solve1(const Matrix<T>&               , int NumEigenValues,const TensorNetworks::Epsilons& eps);
    int Solve1(const PrimeEigenSolverClient<T>*, int NumEigenValues,const TensorNetworks::Epsilons& eps);


    Vector <T> GetEigenVector (int index) const;
    Matrix<T> GetEigenVectors() const
    {
        return itsEigenVectors;
    }
    Vector <double> GetEigenValues () const
    {
        return itsEigenValues;
    }

    static const SparseMatrix<T>* theSparseMatrix;
    static const      Matrix<T>* theDenseMatrix;
protected:
    UdType Solve(primme_params&);

private:
    Vector<double>   itsEigenValues;
    Matrix<T>       itsEigenVectors;
    int              itsNumGuesses;
};

#endif // PRIMEEIGENSOLVER_H
