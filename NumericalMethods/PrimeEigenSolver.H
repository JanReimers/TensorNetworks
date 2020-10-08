#ifndef PRIMEEIGENSOLVER_H
#define PRIMEEIGENSOLVER_H

#include "Containers/SparseMatrix.H"
#include "TensorNetworks/Typedefs.H"

class Epsilons;
class FullStateImp;
//
//  Solve sparse eigen systems using the PRIMME library
//  For efficiency client code should keep and instance of PrimeEigenSolver alive
//  between iterations so that eigen vector initial guess get used.
//
template <class T> class PrimeEigenSolver
{
        typedef TensorNetworks::Matrix4T Matrix4T;
    public:
        PrimeEigenSolver();
        ~PrimeEigenSolver();

        // Get lowest N eigen values and vectors with no initial guess
        int Solve(const DMatrix<T>&                              , int NumEigenValues,const Epsilons& eps);
        int Solve(const Matrix4T& Hlocal, const FullStateImp* Psi, int NumEigenValues,const Epsilons& eps);


        Vector <T> GetEigenVector (int index) const;
        DMatrix<T> GetEigenVectors() const {return itsEigenVectors;}
        Vector <double> GetEigenValues () const {return itsEigenValues;}

        static const SparseMatrix<T>* theSparseMatrix;
        static const      DMatrix<T>* theDenseMatrix;
    protected:
        int SolveSparse(int NumEigenValues,double eps);
        int SolveDense (int NumEigenValues,double eps);


    private:
        Vector<double>   itsEigenValues;
        DMatrix<T>       itsEigenVectors;
        int              itsNumGuesses;
};

#endif // PRIMEEIGENSOLVER_H