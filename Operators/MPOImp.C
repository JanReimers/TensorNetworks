#include "Operators/MPOImp.H"
#include "Operators/SiteOperatorImp.H"
#include "TensorNetworksImp/SVMPOCompressor.H"
#include "TensorNetworks/CheckSpin.H"

namespace TensorNetworks
{

MPOImp::MPOImp(int L, double S,LoadWith loadWith)
    : itsL(L)
    , itsS(S)
    , areSitesLinked(false)
    , itsSites()
{
    assert(isValidSpin(S));
    int d=2*S+1;
    assert(itsL>1);
    assert(d>1);

    switch (loadWith)
    {
    case Identity:
    {
        //
        //  Load up the sites with unit operators
        //  TODO tidy up loop and unit test.
        //
        Insert(new SiteOperatorImp(d,PLeft));
        for (int ia=2; ia<=itsL-1; ia++)
            Insert(new SiteOperatorImp(d,PBulk));
        Insert(new SiteOperatorImp(d,PRight));
        LinkSites();
        break;
    }
    case LoadLater:
    {
        break;
    }
    }

}

MPOImp::MPOImp(int L, double S, const TensorT& W)
    : itsL(L)
    , itsS(S)
    , areSitesLinked(false)
    , itsSites()
{
    assert(isValidSpin(S));
    int d=2*S+1;
    assert(itsL>=1);
    assert(d>1);
    //
    //  Load up the sites with copies of the W operator
    //
    for (int ia=1; ia<=itsL; ia++)
    {
        Position lbr = ia==1 ? PLeft : (ia==L ? PRight : PBulk);
        Insert(new SiteOperatorImp(d,lbr,W));
    }
}



MPOImp::~MPOImp()
{
    //dtor
}

void MPOImp::ConvertToiMPO(int UnitCell)
{
    assert((itsL-UnitCell)%2==0); //(itsL-UnitCell) needs to be even.
    int n=(itsL-UnitCell)/2; //Strip away this many sites from each end.
    //
    //  This ends up being a #$*#(*$^# mess because ptr_vector doesn't support revers iterators.
    //
    for (int in=1; in<=n; in++)
    {
        auto is=itsSites.begin();
        is++; // skip dummy site a index [0]
        itsSites.erase(is);
    }
    for (int in=1; in<=n; in++)
    {
        auto is=itsSites.begin();
        for (int i=0; i<=UnitCell; i++,is++);
        itsSites.erase(is);
    }
    itsL=UnitCell;
    //
    //  Link first and last sites.
    //
    SiteOperatorImp* s=dynamic_cast<SiteOperatorImp*>(itsSites[1]);
    assert(s);
    s->SetNeighbours(itsSites[itsL  ],itsSites[2]);
    s=dynamic_cast<SiteOperatorImp*>(itsSites[itsL]);
    assert(s);
    s->SetNeighbours(itsSites[itsL-1],itsSites[1]);

}

void MPOImp::Insert(SiteOperator* so)
{
    assert(so);
    assert(static_cast<int>(itsSites.size())<=itsL);
    assert(!areSitesLinked);
    if (itsSites.size()==0) itsSites.push_back(0); //Dummy at index 0 so we start counting at index 1
    itsSites.push_back(so);
//  This is too clever by half, derived constructor should call LinkSites.
//    if (static_cast<int>(itsSites.size())==itsL+1) LinkSites(); //Dummy at index 0 so we start counting at index 1
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
    areSitesLinked=true;
}


} //namespace
