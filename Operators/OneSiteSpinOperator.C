#include "Operators/OneSiteSpinOperator.H"
#include "TensorNetworksImp/SpinCalculator.H"

namespace TensorNetworks
{


OneSiteSpinOperator::OneSiteSpinOperator(double S, SpinOperator o)
: itsS(S)
, itsd(2*S+1)
, itsOperator(o)
, itsDw12(1,1,Vector<int>(1),Vector<int>(1))
{
    itsDw12.w1_first(1)=1;
    itsDw12.w2_last (1)=1;
}

OneSiteSpinOperator::~OneSiteSpinOperator()
{
}

MatrixRT OneSiteSpinOperator::GetW (Position lbr,int m, int n) const
{
    MatrixRT W(1,1);
    SpinCalculator sc(itsS);

    switch(itsOperator)
    {
        case Sx:
        {
            W(1,1)=sc.GetSx(m,n);
            break;
        }
        case Sy:
        {
            // THis return a pure imaginary matrix whihc we don;t support yet
            //W(1,1)=sc.GetSy(m,n);
            assert(false);
            break;
        }
        case Sz:
        {
            W(1,1)=sc.GetSz(m,n);
            break;
        }
        case Sp:
        {
            W(1,1)=sc.GetSp(m,n);
            break;
        }
        case Sm:
        {
            W(1,1)=sc.GetSm(m,n);
            break;
        }
    }
    return W;
}

}
