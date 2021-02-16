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
    OperatorElement(int fil); //Required by mat mul routines
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

//
//  OperatorElement/scalar mixing is allowed in oml matrix ops.
//
template <class T> struct ReturnType<T ,TensorNetworks::OperatorElement<T>>
{
    typedef TensorNetworks::OperatorElement<T> RetType;
};
template <class T> struct ReturnType<TensorNetworks::OperatorElement<T>,T >
{
    typedef TensorNetworks::OperatorElement<T> RetType;
};
template <class T> struct ReturnType<TensorNetworks::OperatorElement<T>,TensorNetworks::OperatorElement<T> >
{
    typedef TensorNetworks::OperatorElement<T> RetType;
};

template <class T> double fabs(const TensorNetworks::OperatorElement<T>& oe)
{
    const Matrix<T>& m(oe);
    return Max(fabs(m));
}


#endif // OPERATORELEMENT_H
