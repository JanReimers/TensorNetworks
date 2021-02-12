#ifndef OPERATORVALUEDMATRIX_H
#define OPERATORVALUEDMATRIX_H

#include "Operators/OperatorElement.H"
#include "TensorNetworks/Namespace.H"

namespace TensorNetworks
{

template <class T> class MatrixO : public Matrix<OperatorElement<T> >
{
    typedef Matrix<OperatorElement<T> > Base;
public:
    MatrixO();
    MatrixO(const Base   & m);
    MatrixO(const MatrixO& m);
    MatrixO(MatrixO&& m);
    MatrixO(int Dw1, int Dw2,double S);
    virtual ~MatrixO();

    void CheckUL();
    MatrixO& operator=(const MatrixO& m);
    MatrixO& operator=(MatrixO&& m);

    TriType GetUpperLower() const {return itsUL;}

    MatrixO GetV(Direction) const;
    void    SetV(const MatrixO& V); //Limits of V decide R/L*U/L

private:

    TriType itsUL; //Upper or lower triangular.
};

typedef MatrixO<double> MatrixOR;
typedef MatrixO<dcmplx> MatrixOC;

} //namespace

#endif // OPERATORVALUEDMATRIX_H
