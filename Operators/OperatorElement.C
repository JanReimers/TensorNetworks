#include "Operators/OperatorElement.H"
#include "TensorNetworksImp/SpinCalculator.H"

namespace TensorNetworks
{

template <class T> OperatorElement<T>::OperatorElement(int d,double S)
: Matrix<T>(0,d-1,0,d-1)
, itsS(S)
, itsd(2*S+1)
{
    Fill(*this,T(0.0));
    //ctor
}

template <class T> OperatorElement<T>::OperatorElement()
: Matrix<T>()
, itsS(0.0)
, itsd(0)
{
}

template <class T> OperatorElement<T>::OperatorElement(int)
: Matrix<T>()
, itsS(0.0)
, itsd(0)
{
}

template <class T> OperatorElement<T>::~OperatorElement()
{
    //dtor
}

 OperatorI::OperatorI(double S)
 : OperatorElement(2*S+1,S)
 {
     for (int m=0;m<itsd;m++)
     for (int n=0;n<itsd;n++)
        (*this)(m,n)= (m==n) ? 1.0 : 0.0;
 }

OperatorZ::OperatorZ(double S)
 : OperatorElement(2*S+1,S)
 {
     Fill(*this,0.0);
 }


 OperatorSz::OperatorSz(double S)
 : OperatorElement(2*S+1,S)
 {
     SpinCalculator sc(S);
     for (int m=0;m<itsd;m++)
     for (int n=0;n<itsd;n++)
        (*this)(m,n)=sc.GetSz(m,n);
 }

OperatorSy::OperatorSy(double S)
 : OperatorElement(2*S+1,S)
 {
     SpinCalculator sc(S);
     for (int m=0;m<itsd;m++)
     for (int n=0;n<itsd;n++)
        (*this)(m,n)=sc.GetSy(m,n);
 }

OperatorSx::OperatorSx(double S)
 : OperatorElement(2*S+1,S)
 {
     SpinCalculator sc(S);
     for (int m=0;m<itsd;m++)
     for (int n=0;n<itsd;n++)
        (*this)(m,n)=sc.GetSx(m,n);
 }

OperatorSp::OperatorSp(double S)
 : OperatorElement(2*S+1,S)
 {
     SpinCalculator sc(S);
     for (int m=0;m<itsd;m++)
     for (int n=0;n<itsd;n++)
        (*this)(m,n)=sc.GetSp(m,n);
 }

OperatorSm::OperatorSm(double S)
 : OperatorElement(2*S+1,S)
 {
     SpinCalculator sc(S);
     for (int m=0;m<itsd;m++)
     for (int n=0;n<itsd;n++)
        (*this)(m,n)=sc.GetSm(m,n);
 }

 template class OperatorElement<double>;

 } //Tensor networks
