#ifndef OPERATORELEMENT_H
#define OPERATORELEMENT_H

#include "oml/matrix.h"

namespace TensorNetworks
{

using dcmplx=std::complex<double>;

template <class T> class OperatorElement : public Matrix<T>
{
public:
    OperatorElement(int d,double S);
    virtual ~OperatorElement();

protected:
    double itsS;
    int    itsd;
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
