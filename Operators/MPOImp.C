#include "Operators/MPOImp.H"
#include "Operators/SiteOperatorImp.H"
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
    assert(itsL>1); //One won't work because of the Left/Right boundary sites
    assert(Getd()>1);
}

MPOImp::MPOImp(int L, double S,LoadWith loadWith)
    : MPOImp(L,S)
{
    int d=Getd();
    assert(loadWith==Identity);

    for (int ia=1; ia<=itsL; ia++)
        Insert(new SiteOperatorImp(d));
    LinkSites();

}

MPOImp::MPOImp(int L, const OperatorClient* W)
    : MPOImp(L,W->GetS())
{
    int d=Getd();
    Insert(new SiteOperatorImp(d,PLeft,W));
    for (int ia=2;ia<=GetL()-1;ia++)
        Insert(new SiteOperatorImp(d,PBulk,W));
    Insert(new SiteOperatorImp(d,PRight,W));
    LinkSites();

}

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
