#ifndef OPERATORVALUEDMATRIX_H
#define OPERATORVALUEDMATRIX_H

#include "Operators/OperatorElement.H"
#include "TensorNetworks/Namespace.H"

namespace TensorNetworks
{

template <class T> class MatrixO : public Matrix<OperatorElement<T> >
{
    public:
        MatrixO(const OperatorClient1*);
        MatrixO(int Dw1, int Dw2,double S);
        virtual ~MatrixO();

    protected:

    private:
};

typedef MatrixO<double> MatrixOR;
typedef MatrixO<dcmplx> MatrixOC;

} //namespace

#endif // OPERATORVALUEDMATRIX_H
