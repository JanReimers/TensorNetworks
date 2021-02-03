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
//
// Construct with W operator
//
SiteOperatorLeft::SiteOperatorLeft(int d, const TensorT& W)
    : SiteOperatorImp(d,W)
{
    Init_lr(SiteOperatorImp::itsDw.Dw1);
}


SiteOperatorLeft::~SiteOperatorLeft()
{
    //dtor
}

void SiteOperatorLeft::Init_lr(int oneIndex)
{
    MatrixRT itsl(1,SiteOperatorImp::itsDw.Dw1);
    Fill(itsl,0.0);
    itsl(1,oneIndex)=1.0;

    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            MatrixRT W=GetW(m,n);
            itsWs(m+1,n+1)=itsl*W;
        }
     itsDw.Dw1=1;
     SetLimits();
}

void SiteOperatorLeft::CheckDws() const
{
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

MatrixRT SiteOperatorLeft::GetV(Direction lr,int m, int n) const
{
    MatrixRT V(1,itsDw.Dw2-1);
    const MatrixRT& W=GetW(m,n);
    switch (lr)
    {
    case DLeft:
        for (int w2=2;w2<=itsDw.Dw2;w2++)
            V(1,w2-1)=W(1,w2);
        break;
    case DRight:
        for (int w2=1;w2<=itsDw.Dw2-1;w2++)
            V(1,w2)=W(1,w2);
        break;
    }
    return V;
}

//
//  Load row matrix starting at second index
//
void SiteOperatorLeft::SetV (Direction lr,int m, int n, const MatrixRT& V)
{
    assert(V.GetNumRows()==1);
    MatrixRT& W=itsWs(m+1,n+1);
    assert(W.GetNumRows()==1);
    assert(W.GetNumCols()==V.GetNumCols()+1);

    switch (lr)
    {
    case DLeft:
        for (int w2=2;w2<=itsDw.Dw2;w2++)
            W(1,w2)=V(1,w2-1);
        break;
    case DRight:
        for (int w2=1;w2<=itsDw.Dw2-1;w2++)
            W(1,w2)=V(1,w2);
        break;
    }
}

} //namespace
#define TYPE Matrix<double>
#include "oml/src/matrix.cpp"
