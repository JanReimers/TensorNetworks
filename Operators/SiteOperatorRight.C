#include "Operators/SiteOperatorRight.H"

namespace TensorNetworks
{

//
//  Build with Dw=1 identity operators
//
SiteOperatorRight::SiteOperatorRight(int d)
    : SiteOperatorImp(d)
    , itsDw(1,1)
    , itsWrs(d,d)
{
    Init_lr();
}

SiteOperatorRight::SiteOperatorRight(int d, double S, SpinOperator so) //Construct spin operator
    : SiteOperatorImp(d,S,so)
    , itsDw(1,1)
    , itsWrs(d,d)
{
    Init_lr();
}
//
//  Build from a W rep object
//
SiteOperatorRight::SiteOperatorRight(int d, const OperatorClient* H)
    : SiteOperatorImp(d,H)
    , itsDw(1,1)
    , itsWrs(d,d)
{
    Init_lr();
}


//
// Build from a trotter decomp.
//
SiteOperatorRight::SiteOperatorRight(int d, Direction lr,const MatrixRT& U, const DiagonalMatrixRT& s)
    : SiteOperatorImp(d,lr,U,s)
    , itsDw(1,1)
    , itsWrs(d,d)
{
    Init_lr();
}
//
// Construct with W operator
//
SiteOperatorRight::SiteOperatorRight(int d, const TensorT& W)
    : SiteOperatorImp(d,W)
    , itsDw(1,1)
    , itsWrs(W)
{
    Init_lr();
}


SiteOperatorRight::~SiteOperatorRight()
{
    //dtor
}

void SiteOperatorRight::Init_lr()
{
    assert( itsDw.Dw2==1);
    if (isShapeDirty)
    {
        itsDw.Dw1=SiteOperatorImp::itsDw.Dw1;
        itsr.SetLimits(SiteOperatorImp::itsDw.Dw2,1);
        Fill(itsr,0.0);
        itsr(1,1)=1.0;
        isShapeDirty=false;
    }
    if (isData_Dirty)
    {
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
                itsWrs(m+1,n+1)=itsWs(m+1,n+1)*itsr;
        SetLimits();
        isData_Dirty=true;
    }
    CheckDws();
}

void SiteOperatorRight::CheckDws() const
{
    assert(itsr.GetLimits()==MatLimits(SiteOperatorImp::itsDw.Dw2,1));
#ifdef DEBUG
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            const MatrixRT& W=GetW(m,n);
            assert(W.GetNumCols()==1);
            assert(W.GetNumRows()==itsDw.Dw1);
        }
#endif
    SiteOperatorImp::CheckDws();
}

void SiteOperatorRight::SetLimits()
{
    SiteOperatorImp::SetLimits(itsDw,itsWrs);
    SiteOperatorImp::SetLimits();
}




} //namespace
#define TYPE Matrix<double>
#include "oml/src/matrix.cpp"
