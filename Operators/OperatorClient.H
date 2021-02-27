#ifndef OPERATORCLIENT_H
#define OPERATORCLIENT_H

#include "TensorNetworks/Namespace.H"
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

    virtual double    GetS  () const=0;
    virtual Matrix4RT GetH12() const  ; //Two site interaction matrix
    virtual MatrixOR  GetW  (MPOForm) const=0; //w1, w2 are zero based.
private:
    virtual double    GetH   (int ma,int na,int mb,int nb) const=0;
};

}
#endif // OPERATORCLIENT_H
