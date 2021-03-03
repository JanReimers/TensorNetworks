#include "Operators/MPOImp.H"
#include "Operators/SiteOperatorImp.H"
#include "Operators/OperatorBond.H"
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
        Insert(new SiteOperatorImp(d,FUnit));
    LinkSites();

}


MPOImp::MPOImp(int L, const OperatorClient* W,MPOForm f)
    : MPOImp(L,W->GetS())
{
    int d=Getd();
    for (int ia=1;ia<=GetL();ia++)
        Insert(new SiteOperatorImp(d,GetPosition(L,ia),W,f));
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
    //
    //  Create bond objects
    //
    itsBonds.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    for (int i=1; i<itsL; i++)
    {
        auto [Dw1,Dw2]=itsSites[i]->GetDws();
        itsBonds.push_back(new OperatorBond(Dw2));
    }

    SiteOperatorImp* s=dynamic_cast<SiteOperatorImp*>(itsSites[1]);
    assert(s);
    if (itsL>1)
    {
        s->SetNeighbours(0,itsBonds[1]);
        for (int ia=2; ia<=itsL-1; ia++)
        {
            s=dynamic_cast<SiteOperatorImp*>(itsSites[ia]);
            assert(s);
            s->SetNeighbours(itsBonds[ia-1],itsBonds[ia]);
        }
        s=dynamic_cast<SiteOperatorImp*>(itsSites[itsL]);
        assert(s);
        s->SetNeighbours(itsBonds[itsL-1],0);
    }
    else
    {
        assert(false); //Trap to figure out if this code path ever gets used.
        s->SetNeighbours(0,0);
    }
    //
    //  Tell each bond about its left and right sites.
    //
    for (int i=1; i<itsL; i++)
        itsBonds[i]->SetSites(itsSites[i],itsSites[i+1]);
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

void   MPOImp::Report(std::ostream& os) const
{
    MPO::Report(os); //List the sites
    os << "  Bond  D   Rank  Entropy   Min(Sv)   SvError " << std::endl;
    for (int ib=1; ib<itsL; ib++)
    {
        os << std::setw(3) << ib << "  ";
        itsBonds[ib]->Report(os);
        os << std::endl;
    }
}

double MPOImp::GetTruncationError() const
{
    double ret=0.0;
    for (int ib=1; ib<itsL; ib++)
        ret+=itsBonds[ib]->GetTruncationError();
    return sqrt(ret);
}


} //namespace
