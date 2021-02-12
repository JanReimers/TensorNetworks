#ifndef OPERATORCLIENT_H
#define OPERATORCLIENT_H

#include "TensorNetworks/Namespace.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworksImp/Typedefs.H"
#include "Operators/OperatorValuedMatrix.H"


namespace TensorNetworks
{
//
//  Derive from this in order to load up your site MPO/W matrices
//
class OperatorClient
{
public:
    OperatorClient() {}
    virtual ~OperatorClient() {}

    virtual double    GetS   () const=0;
    virtual Dw12      GetDw12() const=0;
    virtual Matrix4RT GetH12 () const  ; //Two site interaction matrix
    virtual double    GetH   (int ma,int na,int mb,int nb) const=0;
    virtual MatrixRT  GetW   (int m, int n) const=0;
};


class OperatorClient1
{
public:
    OperatorClient1() {}
    virtual ~OperatorClient1() {}

    virtual double    GetS   () const=0;
    virtual Dw12      GetDw12() const=0;
    virtual Matrix4RT GetH12 () const  ; //Two site interaction matrix
    virtual double    GetH   (int ma,int na,int mb,int nb) const=0;

    virtual MatrixOR GetMatrixO(TriType ul) const=0; //w1, w2 are zero based.
};


// Unit operator k-delta function
inline double I(int m, int n)
{
    assert(m>=0);
    assert(n>=0);
    double ret=0.0;
    if (n==m) ret=1.0;
    return ret;
}

}
#endif // OPERATORCLIENT_H
