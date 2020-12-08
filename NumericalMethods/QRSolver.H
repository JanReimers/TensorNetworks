#ifndef QRSOLVER_H
#define QRSOLVER_H

#include <tuple>

template <class T> class Matrix;
//
//  Abstract interface for QR solvers
//
template <class T> class QRSolver
{
protected:
    typedef Matrix<T>                   MatrixT;
    typedef std::tuple<MatrixT,MatrixT> QRType;
public:
    virtual ~QRSolver() {};

//    virtual QRType  Solve    (const MatrixT&,double eps)=0;
    virtual QRType  SolveThin(const MatrixT&)=0;

};

#endif // QRSOLVER_H
