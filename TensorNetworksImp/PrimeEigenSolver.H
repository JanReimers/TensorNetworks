#ifndef PRIMEEIGENSOLVER_H
#define PRIMEEIGENSOLVER_H

#include "Containers/SparseMatrix.H"
//
//  Solve sparse eigen systems using the PRIMME library
//
template <class T> class PrimeEigenSolver
{
    public:
        PrimeEigenSolver(const DMatrix<T>&, double eps);
        ~PrimeEigenSolver();

        // Get lowest N eigen values and vectors with no initial guess
        void Solve(int NumEigenValues);
        // Get lowest eigen value and vector with initial guess
        void SolveLowest(const Vector<T>& EigenVectorGuess);

        Vector <T> GetEigenVector (int index) const;
        DMatrix<T> GetEigenVectors() const {return itsEigenVectors;}
        Vector <double> GetEigenValues () const {return itsEigenValues;}

        static const SparseMatrix<T>* theSparseMatrix;
        static const      DMatrix<T>* theDenseMatrix;

    protected:
        void SolveSparse(int NumEigenValues);
        void SolveDense(int NumEigenValues);


    private:
        Vector<double>   itsEigenValues;
        DMatrix<T>       itsEigenVectors;
        double           itsEps;  //Convergence criteria
        bool             itsChooseSparse;
        SparseMatrix<T>  itsSparsem;
};

#endif // PRIMEEIGENSOLVER_H
