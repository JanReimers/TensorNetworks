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

    virtual QRType  SolveThinQR(const MatrixT&)=0; // returns [Q,R]
    virtual QRType  SolveThinRQ(const MatrixT&)=0; // mirror image version returns [R,Q]
    virtual QRType  SolveThinQL(const MatrixT&)=0; // returns [Q,L]
    virtual QRType  SolveThinLQ(const MatrixT&)=0; // mirror image version returns [L,Q]

};

#endif // QRSOLVER_H
