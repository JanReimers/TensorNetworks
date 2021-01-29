#include "Operators/iMPOImp.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworksImp/StateIterator.H"
#include "Operators/SiteOperatorBulk.H"
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
    , areSitesLinked(false)
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
        Insert(new SiteOperatorBulk(Getd()));

    LinkSites();
}

//
//  Load up the sites with copies of the W operator
//
iMPOImp::iMPOImp(int L, double S,const TensorT& W)
    : iMPOImp(L,S)
{
    for (int ia=1; ia<=itsL; ia++)
        Insert(new SiteOperatorBulk(Getd(),W));
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
        Insert(new SiteOperatorBulk(Getd(),W));
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
    areSitesLinked=true;
}

void iMPOImp::Insert(SiteOperator* so)
{
    assert(so);
    assert(static_cast<int>(itsSites.size())<=itsL);
    assert(!areSitesLinked);
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
    assert(isite<=itsL);
    assert(areSitesLinked);
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
    //  Usually unitcell==L, but maybe we can have 2 or more unit cells fit exaclty inside L.
    //
    assert(unitcell<=L);
    assert(L%unitcell==0);  //Make sure unit cell and L are compatible.
    int newL=L/unitcell;
    assert(newL*unitcell==L); //THis should be the same as the L%unitcell==0 test above.
    int d=iMPO::GetSiteOperator(1)->Getd();

    //
    //  Work out the dimension of the unit cell Hilbert space.
    //
    int newd=1;
    for (int ia=1;ia<=unitcell;ia++)
    {
        assert(iMPO::GetSiteOperator(ia)->Getd()==d); //Assume all site have the same d
        newd*=d;
    }
    double newS=(newd-1)/2.0;
    //
    //  Create a tensor to store W
    //
    TensorT Wcell(newd,newd);

    for (StateIterator m(unitcell,d); !m.end(); m++)
        for (StateIterator n(unitcell,d); !n.end(); n++)
        {
            const Vector<int>& ms=m.GetQuantumNumbers();
            const Vector<int>& ns=n.GetQuantumNumbers();

            MatrixRT W=iMPO::GetSiteOperator(1)->GetW(ms(unitcell),ns(unitcell));
            for (int ia=2;ia<=unitcell;ia++)
                W*=iMPO::GetSiteOperator(ia)->GetW(ms(unitcell-ia+1),ns(unitcell-ia+1));
            int mab=m.GetLinearIndex(); //one based
            int nab=n.GetLinearIndex();
//            cout << "mab, ms=" << mab << " " << ms << endl;
            Wcell(mab,nab)=W;
        }

    return new iMPOImp(newL,newS,Wcell);

}

} //namespace

//---------------------------------------------------------------------------------
//
//  Make template instance.  For some reason we can's share the instance created in SiteOperatorImp.C
//
#define TYPE Matrix<double>
#include "oml/src/matrix.cpp"

