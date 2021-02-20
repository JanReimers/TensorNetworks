#include "Operators/SiteOperatorLeft.H"

namespace TensorNetworks
{

//
//  Build with Dw=1 identity operators
//
SiteOperatorLeft::SiteOperatorLeft(int d)
    : SiteOperatorImp(d)
{
    Init_lr(SiteOperatorImp::itsDw.Dw1);
}

SiteOperatorLeft::SiteOperatorLeft(int d, double S, SpinOperator so) //Construct spin operator
    : SiteOperatorImp(d,S,so)
{
    Init_lr(SiteOperatorImp::itsDw.Dw1);
}


//
//  Build from a W rep object
//
SiteOperatorLeft::SiteOperatorLeft(int d, const OperatorClient* H)
    : SiteOperatorImp(d,H)
{
    Init_lr(SiteOperatorImp::itsDw.Dw1);
}


//
// Build from a trotter decomp.
//
SiteOperatorLeft::SiteOperatorLeft(int d, Direction lr,const MatrixRT& U, const DiagonalMatrixRT& s)
    : SiteOperatorImp(d,lr,U,s)
{
    Init_lr(1);
}

SiteOperatorLeft::~SiteOperatorLeft()
{
    //dtor
}

void SiteOperatorLeft::Init_lr(int oneIndex)
{
    MatrixRT l(0,0,0,SiteOperatorImp::itsDw.Dw1-1);
    Fill(l,0.0);
    l(0,oneIndex-1)=1.0;

    itsWOvM=MatrixOR(l*itsWOvM);
    itsDw.Dw1=1;
    SyncOtoW();
    itsWOvM.SetUpperLower(Lower);
}

} //namespace
