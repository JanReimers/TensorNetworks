#include "TensorNetworksImp/IdentityOperator.H"
#include "TensorNetworks/Dw12.H"


IdentityOperator::IdentityOperator()
{
    Vector<int> Dw2(1);
    Dw2(1)=1;

    itsDw12=new Dw12(1,Dw2);
}

IdentityOperator::~IdentityOperator()
{
    delete itsDw12;
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
