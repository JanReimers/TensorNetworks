#include "Operators/MPO_LRB.H"
#include "Operators/SiteOperatorImp.H"

MPO_LRB::MPO_LRB(int L, double S)
    : itsL(L)
    , itsp(2*S+1)
{
#ifdef DEBUG
    double ipart;
    double frac=std::modf(2.0*S,&ipart);
    assert(frac==0.0);
#endif
    assert(S>=0.5);
}

MPO_LRB::MPO_LRB(const OperatorWRepresentation* O,int L, double S)
    : itsL(L)
    , itsp(2*S+1)
{
#ifdef DEBUG
    double ipart;
    double frac=std::modf(2.0*S,&ipart);
    assert(frac==0.0);
#endif
    assert(S>=0.5);
    Init(O);
}

MPO_LRB::~MPO_LRB()
{
    //dtor
}

void MPO_LRB::Init(const OperatorWRepresentation* O)
{
    assert(O);
    //
    //  Load W matrices for the left edge,bulk and right edge
    //
    SiteOperator* left =new SiteOperatorImp(TensorNetworks::PLeft,O,itsp);
    SiteOperator* bulk =new SiteOperatorImp(TensorNetworks::PBulk,O,itsp);
    SiteOperator* right=new SiteOperatorImp(TensorNetworks::PRight,O,itsp);
    itsSites.push_back(left );     //Left edge
    itsSites.push_back(bulk );     //bulk
    itsSites.push_back(right);     //Right edge
}

TensorNetworks::Position MPO_LRB::GetPosition(int isite) const
{
    return isite==0 ? TensorNetworks::PLeft :
           (isite==itsL-1 ? TensorNetworks::PRight : TensorNetworks::PBulk);
}

