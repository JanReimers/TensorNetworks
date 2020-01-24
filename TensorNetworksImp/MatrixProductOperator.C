#include "TensorNetworksImp/MatrixProductOperator.H"
#include "TensorNetworksImp/SiteOperatorImp.H"

MatrixProductOperator::MatrixProductOperator(int L, int S2)
    : itsL(L)
    , itsp(S2+1)
{
}

MatrixProductOperator::MatrixProductOperator(const OperatorWRepresentation* O,int L, int S2)
    : itsL(L)
    , itsp(S2+1)
{
    Init(O);
}

MatrixProductOperator::~MatrixProductOperator()
{
    //dtor
}

void MatrixProductOperator::Init(const OperatorWRepresentation* O)
{
    assert(O);
    //
    //  Load W matrices for the left edge,bulk and right edge
    //
    SiteOperator* left =new SiteOperatorImp(TensorNetworks::Left,O,itsp);
    SiteOperator* bulk =new SiteOperatorImp(TensorNetworks::Bulk,O,itsp);
    SiteOperator* right=new SiteOperatorImp(TensorNetworks::Right,O,itsp);
    itsSites.push_back(left );     //Left edge
    itsSites.push_back(bulk );     //bulk
    itsSites.push_back(right);     //Right edge
}

TensorNetworks::Position MatrixProductOperator::GetPosition(int isite) const
{
    return isite==0 ? TensorNetworks::Left :
           (isite==itsL-1 ? TensorNetworks::Right : TensorNetworks::Bulk);
}

