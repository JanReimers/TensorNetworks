#include "OperatorValuedMatrix.H"
#include "Operators/OperatorClient.H"

namespace TensorNetworks
{

template <class T> MatrixO<T>::MatrixO(const OperatorClient1* oc)
: Matrix<OperatorElement<T> >(oc->GetMatrixO())
{
    //ctor
}

template <class T> MatrixO<T>::MatrixO(int Dw1, int Dw2,double S)
: Matrix<OperatorElement<T> >(0,Dw1-1,0,Dw2-1)
{
    OperatorElement<T> Z=OperatorZ(S);
    Fill(*this,Z);
}

template <class T> MatrixO<T>::~MatrixO()
{
    //dtor
}



template class MatrixO<double>;

} //namespace

#include "oml/src/matrix.cpp"
#define Type TensorNetworks::OperatorElement<double>

template class Matrix<Type>;
