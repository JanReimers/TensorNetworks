#ifndef OPERATORELEMENT_H
#define OPERATORELEMENT_H

#include "oml/matrix.h"

namespace TensorNetworks
{

using dcmplx=std::complex<double>;

template <class T> class OperatorElement : public Matrix<T>
{
public:
    OperatorElement();
    OperatorElement(int d,double S);
    virtual ~OperatorElement();

    template <class A, Store M, Data D> const OperatorElement& operator=(const Indexable <T,A,M,D,MatrixShape>& m)
    {
        Matrix<T>::operator=(m);
        return *this;
    }

protected:
    double itsS;
    int    itsd;
};

typedef OperatorElement<double> OperatorElementR;
typedef OperatorElement<dcmplx> OperatorElementC;

// Unit operator
class OperatorI : public OperatorElement<double>
{
public:
    OperatorI(double S);
};

// Zero operator
class OperatorZ : public OperatorElement<double>
{
public:
    OperatorZ(double S);
};

class OperatorSz : public OperatorElement<double>
{
public:
    OperatorSz(double S);
};

class OperatorSy : public OperatorElement<dcmplx>
{
public:
    OperatorSy(double S);
};

class OperatorSx : public OperatorElement<double>
{
public:
    OperatorSx(double S);
};

class OperatorSp : public OperatorElement<double>
{
public:
    OperatorSp(double S);
};

class OperatorSm : public OperatorElement<double>
{
public:
    OperatorSm(double S);
};


} //Tensor networks

#endif // OPERATORELEMENT_H
