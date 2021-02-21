#include "Operators/iMPOImp.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworksImp/StateIterator.H"
#include "Operators/SiteOperatorImp.H"
#include "Operators/OperatorClient.H"
#include "TensorNetworks/CheckSpin.H"

using std::cout;
using std::endl;

namespace TensorNetworks
{
//
//  Root constructor.  All other constructors should delegate to this one.
//
iMPOImp::iMPOImp(int L, double S)
    : itsL(L)
    , itsS(S)
    , itsSites()
{
    assert(isValidSpin(S));
    assert(itsL>0);
    assert(Getd()>1);
}


//
//  Load up the sites with unit operators
//
iMPOImp::iMPOImp(int L, double S,LoadWith loadWith)
    : iMPOImp(L,S)
{
    assert(loadWith==Identity);
    for (int ia=1; ia<=itsL; ia++)
        Insert(new SiteOperatorImp(Getd()));

    LinkSites();
}

//
//  Load up the sites with copies of the W operator
//
iMPOImp::iMPOImp(int L, const MatrixOR& W)
    : iMPOImp(L,W.GetS())
{
    for (int ia=1; ia<=itsL; ia++)
        Insert(new SiteOperatorImp(W));
    LinkSites();
}
//
//  Load up the sites with copies of the W operator
//  For and iMPO the sites at the edge of the unit cell are considered Bulk
//
iMPOImp::iMPOImp(int L, const OperatorClient* W)
    : iMPOImp(L,W->GetS())
{
    for (int ia=1; ia<=itsL; ia++)
        Insert(new SiteOperatorImp(Getd(),PBulk,W));
    LinkSites();
}


iMPOImp::~iMPOImp()
{
    //dtor
}

//
//  Each site needs to know its neighbours in order to
//  carry out SVD tranfers, A[1]->U*s*VT, A=U, s*VT -> Transfered to next site.
//  for iMPO we use PBC (Periodic Boundary Conditions)
//
void iMPOImp::LinkSites()
{
    assert(static_cast<int>(itsSites.size())-1==itsL);
    SiteOperatorImp* s=dynamic_cast<SiteOperatorImp*>(itsSites[1]);
    assert(s);
    if (itsL>1)
    {
        s->SetNeighbours(itsSites[itsL],itsSites[2]);
        for (int ia=2; ia<=itsL-1; ia++)
        {
            s=dynamic_cast<SiteOperatorImp*>(itsSites[ia]);
            assert(s);
            s->SetNeighbours(itsSites[ia-1],itsSites[ia+1]);
        }
        s=dynamic_cast<SiteOperatorImp*>(itsSites[itsL]);
        assert(s);
        s->SetNeighbours(itsSites[itsL-1],itsSites[1]);
    }
    else
    {
        s->SetNeighbours(s,s);
    }
}

void iMPOImp::Insert(SiteOperator* so)
{
    assert(so);
    assert(static_cast<int>(itsSites.size())<=itsL);
    if (itsSites.size()==0) itsSites.push_back(0); //Dummy at index 0 so we start counting at index 1
    itsSites.push_back(so);
}

const SiteOperator* iMPOImp::GetSiteOperator(int isite) const
{
    const iMPO* impo(this);
    return const_cast<iMPO*>(impo)->GetSiteOperator(isite);
}

SiteOperator* iMPOImp::GetSiteOperator(int isite)
{
    assert(isite>0);
    if(isite>itsL)
        isite=(isite-1)%itsL+1; //Allow wrap around since this is a infinite lattice.
    assert(isite<=itsL);
    return itsSites[isite];
}

//
// Contract horizontally to make iMPO for the whole unit cell.
//
iMPO* iMPOImp::MakeUnitcelliMPO(int unitcell) const
{
    int L=GetL();
    assert(L>=1);
    //
    //  Usually unit cell==L, but maybe we can have 2 or more unit cells fit exactly inside L.
    //  We also need to support unit cell=N*L where N is an int.
    //
    int newL=L;
    if (L>unitcell)
    {
        newL=L/unitcell;
        assert(L%unitcell==0);  //Make sure unit cell and L are compatible.
        assert(newL*unitcell==L); //This should be the same as the L%unitcell==0 test above.
    }
    else if (L<unitcell)
    {
        assert(unitcell%L==0);  //Make sure unit cell and L are compatible.

    }
    MatrixOR WW=iMPO::GetSiteOperator(1)->GetW();
    for (int ia=2;ia<=unitcell;ia++)
        WW=HorizontalProduct(WW,iMPO::GetSiteOperator(ia)->GetW());

    return new iMPOImp(newL,WW);
}

} //namespace


