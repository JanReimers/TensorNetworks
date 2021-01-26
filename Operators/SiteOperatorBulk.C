#include "Operators/SiteOperatorBulk.H"

namespace TensorNetworks
{

//
//  Build with Dw=1 identity operators
//
SiteOperatorBulk::SiteOperatorBulk(int d)
    : SiteOperatorImp(d)
{
    Update();
}

SiteOperatorBulk::SiteOperatorBulk(int d, double S, SpinOperator so) //Construct spin operator
    : SiteOperatorImp(d,S,so)
{
    Update();
}
//
//  Build from a W rep object
//
SiteOperatorBulk::SiteOperatorBulk(int d, const OperatorClient* H)
    : SiteOperatorImp(d,H)
{
    Update();
}
//
// Build from a trotter decomp.
//
SiteOperatorBulk::SiteOperatorBulk(int d, Direction lr,const MatrixRT& U, const DiagonalMatrixRT& s)
    : SiteOperatorImp(d,lr,U,s)
{
    Update();
}
//
// Construct with W operator
//
SiteOperatorBulk::SiteOperatorBulk(int d, const TensorT& W)
    : SiteOperatorImp(d,W)
{
    Update();
}


SiteOperatorBulk::~SiteOperatorBulk()
{
    //dtor
}

void SiteOperatorBulk::Update()
{
    isShapeDirty=false;
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


} //namespace
