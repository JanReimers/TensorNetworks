#include "Operators/SiteOperatorLeft.H"

namespace TensorNetworks
{

//
//  Build with Dw=1 identity operators
//
SiteOperatorLeft::SiteOperatorLeft(int d)
    : SiteOperatorImp(d,PLeft)
    , itsDw(1,1)
    , itslWs(d,d)
{
    Init_lr();
}

SiteOperatorLeft::SiteOperatorLeft(int d, double S, SpinOperator so) //Construct spin operator
    : SiteOperatorImp(d,PLeft,S,so)
    , itsDw(1,1,Vector<int>(1),Vector<int>(1))
    , itslWs(d,d)
{
    Init_lr();
}
//
//  Build from a W rep object
//
SiteOperatorLeft::SiteOperatorLeft(int d, const OperatorClient* H)
    : SiteOperatorImp(d,PLeft,H)
    , itsDw(H->GetDw12(PLeft))
    , itslWs(d,d)
{
    Init_lr();
}


//
// Build from a trotter decomp.
//
SiteOperatorLeft::SiteOperatorLeft(int d, Direction lr,const MatrixRT& U, const DiagonalMatrixRT& s)
    : SiteOperatorImp(d,PLeft,lr,U,s)
    , itsDw()
    , itslWs(d,d)
{
    Init_lr();
}
//
// Construct with W operator
//
SiteOperatorLeft::SiteOperatorLeft(int d, const TensorT& W)
    : SiteOperatorImp(d,PLeft,W)
    , itsDw()
    , itslWs(W)
{
    Init_lr();
}


SiteOperatorLeft::~SiteOperatorLeft()
{
    //dtor
}

void SiteOperatorLeft::Init_lr()
{
    assert( itsDw.Dw1==1);
    itsDw.Dw2=SiteOperatorImp::itsDw.Dw2;
    if (isShapeDirty)
    {
        itsl.SetLimits(1,SiteOperatorImp::itsDw.Dw1);
        Fill(itsl,0.0);
        itsl(1,SiteOperatorImp::itsDw.Dw1)=1.0;
        isShapeDirty=false;
    }
    if (isData_Dirty)
    {
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
                itslWs(m+1,n+1)=itsl*itsWs(m+1,n+1);
        SetLimits();
        isData_Dirty=true;
    }
    CheckDws();
}

void SiteOperatorLeft::CheckDws() const
{
    assert(itsl.GetLimits()==MatLimits(1,SiteOperatorImp::itsDw.Dw1));
#ifdef DEBUG
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            const MatrixRT& W=GetW(m,n);
            assert(W.GetNumRows()==1);
            assert(W.GetNumCols()==itsDw.Dw2);
        }
#endif
    SiteOperatorImp::CheckDws();
}

void SiteOperatorLeft::SetLimits()
{
    itsDw.w1_first.SetLimits(itsDw.Dw2);
    itsDw.w2_last .SetLimits(itsDw.Dw1);
    Fill(itsDw.w1_first,1);
    Fill(itsDw.w2_last ,itsDw.Dw2);

}




} //namespace
#define TYPE Matrix<double>
#include "oml/src/matrix.cpp"
