#include "Operators/SiteOperatorRight.H"

namespace TensorNetworks
{

//
//  Build with Dw=1 identity operators
//
SiteOperatorRight::SiteOperatorRight(int d)
    : SiteOperatorImp(d)
{
    Init_lr(1);
}

SiteOperatorRight::SiteOperatorRight(int d, double S, SpinOperator so) //Construct spin operator
    : SiteOperatorImp(d,S,so)
{
    Init_lr(1);
}
//
//  Build from a W rep object
//
SiteOperatorRight::SiteOperatorRight(int d, const OperatorClient* H)
    : SiteOperatorImp(d,H)
{
    Init_lr(1);
}


//
// Build from a trotter decomp.
//
SiteOperatorRight::SiteOperatorRight(int d, Direction lr,const MatrixRT& U, const DiagonalMatrixRT& s)
    : SiteOperatorImp(d,lr,U,s)
{
    Init_lr(1);
}

SiteOperatorRight::~SiteOperatorRight()
{
    //dtor
}

void SiteOperatorRight::Init_lr(int oneIndex)
{
    MatrixRT r(0,SiteOperatorImp::itsDw.Dw2-1,0,0);
    Fill(r,0.0);
    r(oneIndex-1,0)=1.0;

    itsWs=MatrixOR(itsWs*r);
    itsDw.Dw2=1;
    SetLimits();
    itsWs.SetUpperLower(Lower);
}

} //namespace

