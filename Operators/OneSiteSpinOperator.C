#include "Operators/OneSiteSpinOperator.H"
#include "TensorNetworksImp/SpinCalculator.H"



OneSiteSpinOperator::OneSiteSpinOperator(double S, TensorNetworks::SpinOperator o)
: itsS(S)
, itsp(2*S+1)
, itsOperator(o)
, itsDw12(1,1,Vector<int>(1),Vector<int>(1))
{
    itsDw12.w1_first(1)=1;
    itsDw12.w2_last (1)=1;
}

OneSiteSpinOperator::~OneSiteSpinOperator()
{
}

TensorNetworks::MatrixT OneSiteSpinOperator::GetW (TensorNetworks::Position lbr,int m, int n) const
{
    TensorNetworks::MatrixT W(1,1);
    SpinCalculator sc(itsS);

    switch(itsOperator)
    {
        case TensorNetworks::Sx:
        {
            W(1,1)=sc.GetSx(m,n);
            break;
        }
        case TensorNetworks::Sy:
        {
            // THis return a pure imaginary matrix whihc we don;t support yet
            //W(1,1)=sc.GetSy(m,n);
            assert(false);
            break;
        }
        case TensorNetworks::Sz:
        {
            W(1,1)=sc.GetSz(m,n);
            break;
        }
        case TensorNetworks::Sp:
        {
            W(1,1)=sc.GetSp(m,n);
            break;
        }
        case TensorNetworks::Sm:
        {
            W(1,1)=sc.GetSm(m,n);
            break;
        }
    }
    return W;
}
