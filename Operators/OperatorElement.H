#ifndef OPERATORELEMENT_H
#define OPERATORELEMENT_H

#include "TensorNetworks/Enums.H"
#include "oml/matrix.h"


namespace TensorNetworks
{

using dcmplx=std::complex<double>;
template <class T> class MatrixO;

template <class T> class OperatorElement : public Matrix<T>
{
    typedef Matrix<T> Base;
public:
    OperatorElement();
    OperatorElement(T S); //Required by mat mul routines
    OperatorElement(int d); //Fill with 0s
    OperatorElement(int d,T fillValue); //FIll with I*fullValue
    OperatorElement(int d,const Base& m); //Fill with matrix
    virtual ~OperatorElement();

    OperatorElement& operator=(T s);
    template <class A, Store M, Data D> const OperatorElement& operator=(const Indexable <T,A,M,D,MatrixShape>& m)
    {
        Matrix<T>::operator=(m);
        return *this;
    }

    int Getd() const
    {
        assert(IsSquare());
        return Base::GetNumRows();
    }

    bool IsSquare() const {return Base::GetNumRows()==Base::GetNumCols();}
    static OperatorElement Create(SpinOperator,int d);
    friend class MatrixO<T>;

protected:
};

inline int    Stod(double S) {return 2*S+1;}
inline double dtoS(int    d) {return 0.5*(d-1);}

typedef OperatorElement<double> OperatorElementR;
typedef OperatorElement<dcmplx> OperatorElementC;

// Unit operator
class OperatorI : public OperatorElement<double>
{
public:
    OperatorI(double S);
    OperatorI(int d) : OperatorI(dtoS(d)) {};
};

// Zero operator
class OperatorZ : public OperatorElement<double>
{
public:
    OperatorZ(double S);
    OperatorZ(int d) : OperatorZ(dtoS(d)) {};
};

class OperatorSz : public OperatorElement<double>
{
public:
    OperatorSz(double S);
    OperatorSz(int d) : OperatorSz(dtoS(d)) {};
};

class OperatorSy : public OperatorElement<dcmplx>
{
public:
    OperatorSy(double S);
    OperatorSy(int d) : OperatorSy(dtoS(d)) {};
};

class OperatorSx : public OperatorElement<double>
{
public:
    OperatorSx(double S);
    OperatorSx(int d) : OperatorSx(dtoS(d)) {};
};

class OperatorSp : public OperatorElement<double>
{
public:
    OperatorSp(double S);
    OperatorSp(int d) : OperatorSp(dtoS(d)) {};
};

class OperatorSm : public OperatorElement<double>
{
public:
    OperatorSm(double S);
    OperatorSm(int d) : OperatorSm(dtoS(d)) {};
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
