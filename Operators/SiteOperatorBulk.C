#include "Operators/SiteOperatorBulk.H"

namespace TensorNetworks
{

//
//  Build with Dw=1 identity operators
//
SiteOperatorBulk::SiteOperatorBulk(int d)
    : SiteOperatorImp(d)
{
    SetLimits();
}

SiteOperatorBulk::SiteOperatorBulk(int d, double S, SpinOperator so) //Construct spin operator
    : SiteOperatorImp(d,S,so)
{
    SetLimits();
}
//
//  Build from a W rep object
//
SiteOperatorBulk::SiteOperatorBulk(int d, const OperatorClient* H)
    : SiteOperatorImp(d,H)
{
    SetLimits();
}
//
// Build from a trotter decomp.
//
SiteOperatorBulk::SiteOperatorBulk(int d, Direction lr,const MatrixRT& U, const DiagonalMatrixRT& s)
    : SiteOperatorImp(d,lr,U,s)
{
    SetLimits();
}
//
// Construct with W operator
//
SiteOperatorBulk::SiteOperatorBulk(int d, const TensorT& W)
    : SiteOperatorImp(d,W)
{
    SetLimits();
}


SiteOperatorBulk::~SiteOperatorBulk()
{
    //dtor
}

void SiteOperatorBulk::CheckDws() const
{
    SiteOperatorImp::CheckDws();
#ifdef DEBUG
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            const MatrixRT& W=GetW(m,n);
            assert(W.GetNumRows()==itsDw.Dw1);
            assert(W.GetNumCols()==itsDw.Dw2);
        }
#endif
}

MatrixRT SiteOperatorBulk::GetV(Direction lr,int m, int n) const
{
    MatrixRT V(itsDw.Dw1-1,itsDw.Dw2-1);
    const MatrixRT& W=GetW(m,n);
    switch (lr)
    {
    case DLeft:
        for (int w1=2;w1<=itsDw.Dw1;w1++)
            for (int w2=2;w2<=itsDw.Dw2;w2++)
                V(w1-1,w2-1)=W(w1,w2);
        break;
    case DRight:
        for (int w1=1;w1<=itsDw.Dw1-1;w1++)
            for (int w2=1;w2<=itsDw.Dw2-1;w2++)
                V(w1,w2)=W(w1,w2);
        break;
    }
    return V;
}

void SiteOperatorBulk::SetV (Direction lr,int m, int n, const MatrixRT& V)
{
    MatrixRT& W=itsWs(m+1,n+1);
    assert(W.GetNumRows()==V.GetNumRows()+1);
    assert(W.GetNumCols()==V.GetNumCols()+1);
    switch (lr)
    {
    case DLeft:
        for (int w1=2;w1<=itsDw.Dw1;w1++)
            for (int w2=2;w2<=itsDw.Dw2;w2++)
                W(w1,w2)=V(w1-1,w2-1);
        break;
    case DRight:
        for (int w1=1;w1<=itsDw.Dw1-1;w1++)
            for (int w2=1;w2<=itsDw.Dw2-1;w2++)
                W(w1,w2)=V(w1,w2);
        break;
    }
}


} //namespace
