#include "Operators/SiteOperatorBulk.H"

namespace TensorNetworks
{

//
//  Build with Dw=1 identity operators
//
SiteOperatorBulk::SiteOperatorBulk(int d)
    : SiteOperatorImp(d)
{
    Init_lr();
}

SiteOperatorBulk::SiteOperatorBulk(int d, double S, SpinOperator so) //Construct spin operator
    : SiteOperatorImp(d,S,so)
{
    Init_lr();
}
//
//  Build from a W rep object
//
SiteOperatorBulk::SiteOperatorBulk(int d, const OperatorClient* H)
    : SiteOperatorImp(d,H)
{
    Init_lr();
}


//
// Build from a trotter decomp.
//
SiteOperatorBulk::SiteOperatorBulk(int d, Direction lr,const MatrixRT& U, const DiagonalMatrixRT& s)
    : SiteOperatorImp(d,lr,U,s)
{
    Init_lr();
}
//
// Construct with W operator
//
SiteOperatorBulk::SiteOperatorBulk(int d, const TensorT& W)
    : SiteOperatorImp(d,W)
{
    Init_lr();
}


SiteOperatorBulk::~SiteOperatorBulk()
{
    //dtor
}

void SiteOperatorBulk::Init_lr()
{
    if (isData_Dirty)
    {
        SetLimits();
        isData_Dirty=false;
    }
    CheckDws();
}

void SiteOperatorBulk::CheckDws() const
{
    SiteOperatorImp::CheckDws();
}

void SiteOperatorBulk::SetLimits()
{
    itsDw.w1_first.SetLimits(itsDw.Dw2);
    itsDw.w2_last .SetLimits(itsDw.Dw1);
    Fill(itsDw.w1_first,1);
    Fill(itsDw.w2_last ,itsDw.Dw2);

}




} //namespace
