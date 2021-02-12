#include "OperatorValuedMatrix.H"

namespace TensorNetworks
{

template <class T> MatrixO<T>::MatrixO()
: Matrix<OperatorElement<T> >()
, itsUL(Full)
{}

template <class T> MatrixO<T>::MatrixO(const Base& m)
: Matrix<OperatorElement<T> >(m)
, itsUL(Full)
{
    CheckUL();
}

template <class T> MatrixO<T>::MatrixO(const MatrixO& m)
: Matrix<OperatorElement<T> >(m)
, itsUL(Full)
{
    CheckUL();
}

template <class T> MatrixO<T>::MatrixO(MatrixO&& m)
: Matrix<OperatorElement<T> >(m)
, itsUL(Full)
{
    CheckUL();
}


template <class T> MatrixO<T>::MatrixO(int Dw1, int Dw2,double S)
: Matrix<OperatorElement<T> >(0,Dw1-1,0,Dw2-1)
, itsUL(Full)
{
    OperatorElement<T> Z=OperatorZ(S);
    Fill(*this,Z);
}


template <class T> MatrixO<T>::~MatrixO()
{
    //dtor
}
template <class T> void MatrixO<T>::CheckUL()
{
    if (IsLowerTriangular(*this))
        itsUL=Lower;
    else if (IsUpperTriangular(*this))
        itsUL=Upper;
    else
        itsUL=Full;
}
template <class T> MatrixO<T>& MatrixO<T>::operator=(const MatrixO<T>& m)
{
    Base::operator=(m);
    CheckUL();
    return *this;
}
template <class T> MatrixO<T>& MatrixO<T>::operator=(MatrixO<T>&& m)
{
    Base::operator=(m);
    CheckUL();
    return *this;
}


template <class T> MatrixO<T> MatrixO<T>::GetV(Direction lr) const
{
    MatrixO<T> V;
    const MatLimits& l=this->GetLimits();
    MatLimits lv;
    switch (lr)
    {
    case DLeft:
        switch(itsUL)
        {
            case Upper:
                lv=MatLimits(l.Row.Low,l.Row.High-1,l.Col.Low,l.Col.High-1);
            break;
            case Lower:
                lv=MatLimits(l.Row.Low+1,l.Row.High,l.Col.Low+1,l.Col.High);
                break;
            default:
                assert(false);
        }
        break;
    case DRight:
        switch(itsUL)
        {
            case Upper:
                lv=MatLimits(l.Row.Low+1,l.Row.High,l.Col.Low+1,l.Col.High);
            break;
            case Lower:
                lv=MatLimits(l.Row.Low,l.Row.High-1,l.Col.Low,l.Col.High-1);
                break;
            default:
                assert(false);
        }
        break;
    }
    return this->SubMatrix(lv);
}


template class MatrixO <double>;

} //namespace

#include "oml/src/matrix.cpp"
#define Type TensorNetworks::OperatorElement<double>

template class Matrix<Type>;
