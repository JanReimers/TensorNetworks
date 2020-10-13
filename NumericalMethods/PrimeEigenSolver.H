#ifndef PRIMEEIGENSOLVER_H
#define PRIMEEIGENSOLVER_H

#include "TensorNetworksImp/Namespace.H"
#include "Containers/SparseMatrix.H"

struct primme_params;

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
public:
    PrimeEigenSolver();
    ~PrimeEigenSolver();

    // Get lowest N eigen values and vectors with no initial guess
    int Solve(const DMatrix<T>&               , int NumEigenValues,const TensorNetworks::Epsilons& eps);
    int Solve(const PrimeEigenSolverClient<T>*, int NumEigenValues,const TensorNetworks::Epsilons& eps);


    Vector <T> GetEigenVector (int index) const;
    DMatrix<T> GetEigenVectors() const
    {
        return itsEigenVectors;
    }
    Vector <double> GetEigenValues () const
    {
        return itsEigenValues;
    }

    static const SparseMatrix<T>* theSparseMatrix;
    static const      DMatrix<T>* theDenseMatrix;
protected:
    int Solve(primme_params&);

private:
    Vector<double>   itsEigenValues;
    DMatrix<T>       itsEigenVectors;
    int              itsNumGuesses;
};

#endif // PRIMEEIGENSOLVER_H
