#include "Operators/OperatorElement.H"
#include "TensorNetworksImp/SpinCalculator.H"
#include "TensorNetworks/CheckSpin.H"

namespace TensorNetworks
{

template <class T> OperatorElement<T>::OperatorElement(int d)
: Matrix<T>(0,d-1,0,d-1)
{
    Fill(*this,T(0.0));
}

template <class T> OperatorElement<T>::OperatorElement(int d,T fillValue)
: OperatorElement(d)
{
    *this=OperatorI(d)*fillValue;
}

template <class T> OperatorElement<T>::OperatorElement()
: Matrix<T>()
{
}

template <class T> OperatorElement<T>::OperatorElement(T S)
: Matrix<T>()
{
    if (isValidSpin(S))
    {
        int d=Stod(S);
        Matrix<T>::SetLimits(0,d-1,0,d-1);
    }
    else if (S==0.0)
    {
        //S is not a spin. This gets called by oml matrix mul routines.
    }
    else
    {
        std::cerr << "OperatorElement(T S): Illegal spin value S=" << S << std::endl;
        assert(false);
    }
}

template <class T> OperatorElement<T>::~OperatorElement()
{
    //dtor
}

//
// Fill with unit op x s
//
template <class T> OperatorElement<T>& OperatorElement<T>::operator=(T s)
{
    *this=OperatorI(Getd())*s;
    return *this;
}

template <> OperatorElement<double> OperatorElement<double>::Create(SpinOperator so,int d)
{
    OperatorElement<double> O;
        switch (so)
        {
        case Sx:
            O=OperatorSx(d);
            break;
        case Sy:
            assert(false);//O=OperatorSy(S);  THis is a complex operator
            break;
        case Sz:
            O=OperatorSz(d);
            break;
        case Sp:
            O=OperatorSp(d);
            break;
        case Sm:
            O=OperatorSm(d);
            break;
        }
    return O;
}




 OperatorI::OperatorI(double S)
 : OperatorElement(Stod(S))
 {
     for (int m=0;m<Getd();m++)
     for (int n=0;n<Getd();n++)
        (*this)(m,n)= (m==n) ? 1.0 : 0.0;
 }

OperatorZ::OperatorZ(double S)
 : OperatorElement(Stod(S))
 {
     Fill(*this,0.0);
 }


 OperatorSz::OperatorSz(double S)
 : OperatorElement(Stod(S))
 {
     SpinCalculator sc(S);
     for (int m=0;m<Getd();m++)
     for (int n=0;n<Getd();n++)
        (*this)(m,n)=sc.GetSz(m,n);
 }

OperatorSy::OperatorSy(double S)
 : OperatorElement(Stod(S))
 {
     SpinCalculator sc(S);
     for (int m=0;m<Getd();m++)
     for (int n=0;n<Getd();n++)
        (*this)(m,n)=sc.GetSy(m,n);
 }

OperatorSx::OperatorSx(double S)
 : OperatorElement(Stod(S))
 {
     SpinCalculator sc(S);
     for (int m=0;m<Getd();m++)
     for (int n=0;n<Getd();n++)
        (*this)(m,n)=sc.GetSx(m,n);
 }

OperatorSp::OperatorSp(double S)
 : OperatorElement(Stod(S))
 {
     SpinCalculator sc(S);
     for (int m=0;m<Getd();m++)
     for (int n=0;n<Getd();n++)
        (*this)(m,n)=sc.GetSp(m,n);
 }

OperatorSm::OperatorSm(double S)
 : OperatorElement(Stod(S))
 {
     SpinCalculator sc(S);
     for (int m=0;m<Getd();m++)
     for (int n=0;n<Getd();n++)
        (*this)(m,n)=sc.GetSm(m,n);
 }

 template class OperatorElement<double>;

 } //Tensor networks
