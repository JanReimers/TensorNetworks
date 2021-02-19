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

} //namespace
