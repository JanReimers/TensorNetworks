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
    Init_lr(1);
    Update();
}

SiteOperatorRight::SiteOperatorRight(int d, double S, SpinOperator so) //Construct spin operator
    : SiteOperatorImp(d,S,so)
    , itsDw(1,1)
    , itsWrs(d,d)
{
    Init_lr(1);
    Update();
}
//
//  Build from a W rep object
//
SiteOperatorRight::SiteOperatorRight(int d, const OperatorClient* H)
    : SiteOperatorImp(d,H)
    , itsDw(1,1)
    , itsWrs(d,d)
{
    Init_lr(1);
    Update();
}


//
// Build from a trotter decomp.
//
SiteOperatorRight::SiteOperatorRight(int d, Direction lr,const MatrixRT& U, const DiagonalMatrixRT& s)
    : SiteOperatorImp(d,lr,U,s)
    , itsDw(1,1)
    , itsWrs(d,d)
{
    Init_lr(1);
    Update();
}
//
// Construct with W operator
//
SiteOperatorRight::SiteOperatorRight(int d, const TensorT& W)
    : SiteOperatorImp(d,W)
    , itsDw(1,1)
    , itsWrs(W)
{
    Init_lr(1);
    Update();
}


SiteOperatorRight::~SiteOperatorRight()
{
    //dtor
}

void SiteOperatorRight::Init_lr(int oneIndex)
{
    assert( itsDw.Dw2==1);
    itsDw.Dw1=SiteOperatorImp::itsDw.Dw1;
    itsr.SetLimits(SiteOperatorImp::itsDw.Dw2,1);
    Fill(itsr,0.0);
    itsr(oneIndex,1)=1.0;
}

void SiteOperatorRight::Update()
{
    assert( itsDw.Dw2==1);
    assert(itsr.GetNumCols()==1);
    itsDw.Dw1=SiteOperatorImp::itsDw.Dw1;
    if (itsr.GetNumRows()!=SiteOperatorImp::itsDw.Dw2)
    {
        bool OneOne=itsr(1,1)==1.0; //The 1 is in the first element.
        if (itsr.size()==1) //THis is hard part
            OneOne=false;

        itsr.SetLimits(1,SiteOperatorImp::itsDw.Dw2);
        Fill(itsr,0.0);
        if (OneOne)
            itsr(1,1)=1.0; //[100...00]
        else
            itsr(SiteOperatorImp::itsDw.Dw2,1)=1.0; //[000...001]
    }
    isShapeDirty=false;
    if (isData_Dirty)
    {
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
                itsWrs(m+1,n+1)=itsWs(m+1,n+1)*itsr;
        itsWs=itsWrs;
        SiteOperatorImp::itsDw=itsDw;
        SetLimits();
        isData_Dirty=false;
    }
    CheckDws();
}


void SiteOperatorRight::CheckDws() const
{
//    assert(itsr.GetLimits()==MatLimits(SiteOperatorImp::itsDw.Dw2,1));
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

void SiteOperatorRight::Product(const SiteOperator* O2)
{
    const SiteOperatorRight* o2=dynamic_cast<const SiteOperatorRight*>(O2);
    assert(o2);
//    std::cout << "itsr=" << itsr << std::endl;
//    std::cout << "o2 itsr=" << o2->itsr << std::endl;
    itsr=TensorProduct(itsr,o2->itsr);
//    std::cout << "itsr=" << itsr << " " << this << std::endl;
    SiteOperatorImp::Product(O2);
}



} //namespace
#define TYPE Matrix<double>
#include "oml/src/matrix.cpp"
