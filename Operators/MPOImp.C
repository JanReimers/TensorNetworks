#include "Operators/MPOImp.H"
#include "Operators/SiteOperatorLeft.H"
#include "Operators/SiteOperatorBulk.H"
#include "Operators/SiteOperatorRight.H"
#include "Operators/OperatorClient.H"
#include "TensorNetworks/CheckSpin.H"

namespace TensorNetworks
{
//
//  Root constructor.  All other constructors should delegate to this one.
//
MPOImp::MPOImp(int L, double S)
    : itsL(L)
    , itsS(S)
    , itsSites()
{
    assert(isValidSpin(S));
    assert(itsL>1);
    assert(Getd()>1);
}

MPOImp::MPOImp(int L, double S,LoadWith loadWith)
    : MPOImp(L,S)
{
    int d=Getd();
    assert(loadWith==Identity);

    Insert(new SiteOperatorLeft(d));
    for (int ia=2; ia<=itsL-1; ia++)
        Insert(new SiteOperatorBulk(d));
    Insert(new SiteOperatorRight(d));
    LinkSites();

}

MPOImp::MPOImp(int L, const OperatorClient* W)
    : MPOImp(L,W->GetS())
{
    int d=Getd();
    Insert(new SiteOperatorLeft(d,W));
    for (int ia=2;ia<=GetL()-1;ia++)
        Insert(new SiteOperatorBulk(d,W));
    Insert(new SiteOperatorRight(d,W));
    LinkSites();

}


//MPOImp::MPOImp(int L, double S, const TensorT& W)
//    : MPOImp(L,S)
//{
//    //
//    //  Load up the sites with copies of the W operator
//    //
//    int d=Getd();
//    for (int ia=1; ia<=itsL; ia++)
//    {
//        Insert(new SiteOperatorBulk(d,W));
//    }
//}



MPOImp::~MPOImp()
{
    //dtor
}

void MPOImp::Insert(SiteOperator* so)
{
    assert(so);
    assert(static_cast<int>(itsSites.size())<=itsL);
    if (itsSites.size()==0) itsSites.push_back(0); //Dummy at index 0 so we start counting at index 1
    itsSites.push_back(so);
}

//
//  Each site needs to know its neighbours in order to
//  carry out SVD tranfers, A[1]->U*s*VT, A=U, s*VT -> Transfered to next site.
//
void MPOImp::LinkSites()
{
    assert(static_cast<int>(itsSites.size())-1==itsL);
    SiteOperatorImp* s=dynamic_cast<SiteOperatorImp*>(itsSites[1]);
    assert(s);
    if (itsL>1)
    {
        s->SetNeighbours(0,itsSites[2]);
        for (int ia=2; ia<=itsL-1; ia++)
        {
            s=dynamic_cast<SiteOperatorImp*>(itsSites[ia]);
            assert(s);
            s->SetNeighbours(itsSites[ia-1],itsSites[ia+1]);
        }
        s=dynamic_cast<SiteOperatorImp*>(itsSites[itsL]);
        assert(s);
        s->SetNeighbours(itsSites[itsL-1],0);
    }
    else
    {
        s->SetNeighbours(0,0);
    }
}

const SiteOperator* MPOImp::GetSiteOperator(int isite) const
{
    const MPO* mpo(this);
    return const_cast<MPO*>(mpo)->GetSiteOperator(isite);
}

SiteOperator* MPOImp::GetSiteOperator(int isite)
{
    assert(isite>0);
    assert(isite<=itsL);
    return itsSites[isite];
}


} //namespace
