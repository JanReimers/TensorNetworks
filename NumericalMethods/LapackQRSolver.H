#ifndef LAPACKQRSOLVER_H
#define LAPACKQRSOLVER_H

#include "NumericalMethods/QRSolver.H"

//
//  Abstract interface for QR solvers
//
template <class T> class LapackQRSolver
    : public virtual QRSolver<T>
{
protected:
    using MatrixT=typename QRSolver<T>::MatrixT;
    using QRType =typename QRSolver<T>::QRType;
public:
    virtual ~LapackQRSolver() {};

//    virtual QRType  Solve    (const MatrixT&);
    virtual QRType  SolveThin(const MatrixT&);

};

#endif // LAPACKQRSOLVER_H
