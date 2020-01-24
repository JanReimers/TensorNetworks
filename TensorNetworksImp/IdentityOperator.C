#include "TensorNetworksImp/IdentityOperator.H"

IdentityOperator::IdentityOperator()
{
    //ctor
}

IdentityOperator::~IdentityOperator()
{
    //dtor
}

TensorNetworks::ipairT  IdentityOperator::GetDw(TensorNetworks::Position) const
{
    return TensorNetworks::ipairT(1,1);
}

TensorNetworks::MatrixT IdentityOperator::GetW (TensorNetworks::Position lbr,int m, int n) const
{
    TensorNetworks::MatrixT W(1,1);
    W(1,1)=m==n ? 1.0 : 0.0;
    return W;
}
