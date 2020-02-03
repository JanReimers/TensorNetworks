#ifndef PRIMEEIGENSOLVER_H
#define PRIMEEIGENSOLVER_H

#include "Containers/SparseMatrix.H"
class Epsilons;
//
//  Solve sparse eigen systems using the PRIMME library
//  For efficiency client code should keep and instance of PrimeEigenSolver alive
//  between iterations so that eigen vector initial guess get used.
//
template <class T> class PrimeEigenSolver
{
    public:
        PrimeEigenSolver(double eps);
        ~PrimeEigenSolver();

        // Get lowest N eigen values and vectors with no initial guess
        void Solve(const DMatrix<T>&, int NumEigenValues,const Epsilons& eps);

        Vector <T> GetEigenVector (int index) const;
        DMatrix<T> GetEigenVectors() const {return itsEigenVectors;}
        Vector <double> GetEigenValues () const {return itsEigenValues;}

        static const SparseMatrix<T>* theSparseMatrix;
        static const      DMatrix<T>* theDenseMatrix;
    protected:
        void SolveSparse(int NumEigenValues,double eps);
        void SolveDense (int NumEigenValues,double eps);


    private:
        Vector<double>   itsEigenValues;
        DMatrix<T>       itsEigenVectors;
        int              itsNumGuesses;
};

#endif // PRIMEEIGENSOLVER_H
