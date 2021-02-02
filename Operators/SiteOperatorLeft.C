#include "Operators/SiteOperatorLeft.H"

namespace TensorNetworks
{

//
//  Build with Dw=1 identity operators
//
SiteOperatorLeft::SiteOperatorLeft(int d)
    : SiteOperatorImp(d)
//    , itsDw(1,1)
    , itslWs(d,d)
{
    Init_lr(SiteOperatorImp::itsDw.Dw1);
    Update();
}

SiteOperatorLeft::SiteOperatorLeft(int d, double S, SpinOperator so) //Construct spin operator
    : SiteOperatorImp(d,S,so)
//    , itsDw(1,1)
    , itslWs(d,d)
{
    Init_lr(SiteOperatorImp::itsDw.Dw1);
    Update();
}


//
//  Build from a W rep object
//
SiteOperatorLeft::SiteOperatorLeft(int d, const OperatorClient* H)
    : SiteOperatorImp(d,H)
//    , itsDw(1,1)
    , itslWs(d,d)
{
    Init_lr(SiteOperatorImp::itsDw.Dw1);
    Update();
}


//
// Build from a trotter decomp.
//
SiteOperatorLeft::SiteOperatorLeft(int d, Direction lr,const MatrixRT& U, const DiagonalMatrixRT& s)
    : SiteOperatorImp(d,lr,U,s)
//    , itsDw(1,1)
    , itslWs(d,d)
{
    Init_lr(1);
    Update();
}
//
// Construct with W operator
//
SiteOperatorLeft::SiteOperatorLeft(int d, const TensorT& W)
    : SiteOperatorImp(d,W)
//    , itsDw(1,1)
    , itslWs(W)
{
    Init_lr(SiteOperatorImp::itsDw.Dw1);
    Update();
}


SiteOperatorLeft::~SiteOperatorLeft()
{
    //dtor
}

void SiteOperatorLeft::Init_lr(int oneIndex)
{
//    assert( itsDw.Dw1==1);
//    itsDw.Dw2=SiteOperatorImp::itsDw.Dw2;
    itsl.SetLimits(1,SiteOperatorImp::itsDw.Dw1);
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

//
//
//
void SiteOperatorLeft::Update()
{
//    assert(itsDw.Dw1==1);
    assert(itsl.GetNumRows()==1);
//    itsDw.Dw2=SiteOperatorImp::itsDw.Dw2;
//    if (itsl.GetNumCols()!=SiteOperatorImp::itsDw.Dw1)
//    {
//        bool OneOne=itsl(1,1)==1.0; //The 1 is in the first element.
//        if (itsl.size()==1) //THis is hard part
//            OneOne=false;
//
//        itsl.SetLimits(1,SiteOperatorImp::itsDw.Dw1);
//        Fill(itsl,0.0);
//        if (OneOne)
//            itsl(1,1)=1.0; //[100...00]
//        else
//            itsl(1,SiteOperatorImp::itsDw.Dw1)=1.0; //[000...001]
//    }
    isShapeDirty=false;
    if (isData_Dirty)
    {
//        for (int m=0; m<itsd; m++)
//            for (int n=0; n<itsd; n++)
//                itslWs(m+1,n+1)=itsl*itsWs(m+1,n+1);
//        itsWs=itslWs;
//        SiteOperatorImp::itsDw.Dw1=1;
        SetLimits();
        isData_Dirty=false;
    }
    CheckDws();
}

void SiteOperatorLeft::CheckDws() const
{
//    assert(itsl.GetLimits()==MatLimits(1,SiteOperatorImp::itsDw.Dw1));
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
//    SiteOperatorImp::SetLimits(itsDw,itslWs);
    SiteOperatorImp::SetLimits();
}

void SiteOperatorLeft::Product(const SiteOperator* O2)
{
    const SiteOperatorLeft* o2=dynamic_cast<const SiteOperatorLeft*>(O2);
    assert(o2);
//    std::cout << "itsl=" << itsl << std::endl;
//    std::cout << "o2 itsl=" << o2->itsl << std::endl;
    itsl=TensorProduct(itsl,o2->itsl);
//    std::cout << "itsl=" << itsl << " " << this << std::endl;
    SiteOperatorImp::Product(O2);
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
    isData_Dirty=true;
}

} //namespace
#define TYPE Matrix<double>
#include "oml/src/matrix.cpp"
