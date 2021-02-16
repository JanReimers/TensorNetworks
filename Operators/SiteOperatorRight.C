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
//
// Construct with W operator
//
SiteOperatorRight::SiteOperatorRight(int d, const TensorT& W)
    : SiteOperatorImp(d,W)
{
    Init_lr(1);
}


SiteOperatorRight::~SiteOperatorRight()
{
    //dtor
}

void SiteOperatorRight::Init_lr(int oneIndex)
{
    MatrixRT itsr(SiteOperatorImp::itsDw.Dw2,1);
    Fill(itsr,0.0);
    itsr(oneIndex,1)=1.0;
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            MatrixRT W=GetW(m,n);
            itsWs(m+1,n+1)=W*itsr;
        }
     itsDw.Dw2=1;
     SetLimits();
     SyncWtoO();
     itsWOvM.SetUpperLower(Lower);
}


void SiteOperatorRight::CheckDws() const
{
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



MatrixRT SiteOperatorRight::GetV(Direction lr,int m, int n) const
{
    MatrixRT V(itsDw.Dw1-1,1);
    const MatrixRT& W=GetW(m,n);
    switch (lr)
    {
    case DLeft:
        for (int w1=2;w1<=itsDw.Dw1;w1++)
            V(w1-1,1)=W(w1,1);
        break;
    case DRight:
        for (int w1=1;w1<=itsDw.Dw1-1;w1++)
            V(w1,1)=W(w1,1);
        break;
    }
    return V;
}

//
//  Load row matrix starting at second index
//
void SiteOperatorRight::SetV (Direction lr,int m, int n, const MatrixRT& V)
{
    assert(V.GetNumCols()==1);
    MatrixRT& W=itsWs(m+1,n+1);
    assert(W.GetNumCols()==1);
    assert(W.GetNumRows()==V.GetNumRows()+1);

    switch (lr)
    {
    case DLeft:
        for (int w1=2;w1<=itsDw.Dw1;w1++)
            W(w1,1)=V(w1-1,1);
        break;
    case DRight:
        for (int w1=1;w1<=itsDw.Dw1-1;w1++)
            W(w1,1)=V(w1,1);
        break;
    }
}


} //namespace
#define TYPE Matrix<double>
#include "oml/src/matrix.cpp"
