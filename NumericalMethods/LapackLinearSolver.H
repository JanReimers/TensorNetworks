#ifndef LAPACKLINEARSOLVER_H
#define LAPACKLINEARSOLVER_H

#include "NumericalMethods/LinearSolver.H"

//
//  Abstract interface for QR solvers
//
template <class T> class LapackLinearSolver
    : public virtual LinearSolver<T>
{
protected:
    using MatrixT=typename LinearSolver<T>::MatrixT;
    using VectorT=typename LinearSolver<T>::VectorT;
public:
    virtual ~LapackLinearSolver() {};
//
//  Non packed triangular systems, solve
//
    virtual VectorT SolveUpperTri(const MatrixT& A,const VectorT& b); // A is upper triangular
    virtual VectorT SolveLowerTri(const MatrixT& A,const VectorT& b); // A is lower triangular
private:
    VectorT SolveTri(const MatrixT& A,const VectorT& b, char UL);
};

#endif // LAPACKLINEARSOLVER_H
