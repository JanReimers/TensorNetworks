#include "Operators/iMPOImp.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworksImp/StateIterator.H"
#include "Operators/SiteOperatorBulk.H"
#include "TensorNetworks/Dw12.H"
#include "oml/vector.h"
#include "oml/matrix.h"
//#include "oml/src/vector.cpp"

using std::cout;
using std::endl;

namespace TensorNetworks
{

iMPOImp::iMPOImp(int L, double S,MPOImp::LoadWith loadWith)
    : MPOImp(L,S,loadWith)
{
    LinkSites();
}

iMPOImp::iMPOImp(int L, double S,const TensorT& W)
    : MPOImp(L,S,W)
{
    LinkSites();
}

iMPOImp::iMPOImp(int L, double S,const OperatorClient* W)
    : MPOImp(L,S,MPOImp::LoadLater)
{
    int d=2*GetS()+1;
    //
    //  Load up the sites with copies of the W operator
    //  For and iMPO the sites at the edge of the unit cell are considered Bulk
    //
    for (int ia=1; ia<=itsL; ia++)
        Insert(new SiteOperatorBulk(d,W));

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
    int d=GetSiteOperator(1)->Getd();

    //
    //  Work out the dimension of the unit cell Hilbert space.
    //
    int newd=1;
    for (int ia=1;ia<=unitcell;ia++)
    {
        assert(GetSiteOperator(ia)->Getd()==d); //Assume all site have the same d
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

            MatrixRT W=GetSiteOperator(1)->GetW(ms(unitcell),ns(unitcell));
            for (int ia=2;ia<=unitcell;ia++)
                W*=GetSiteOperator(ia)->GetW(ms(unitcell-ia+1),ns(unitcell-ia+1));
            int mab=m.GetLinearIndex(); //one based
            int nab=n.GetLinearIndex();
//            cout << "mab, ms=" << mab << " " << ms << endl;
            Wcell(mab,nab)=W;
        }

    return new iMPOImp(newL,newS,Wcell);

}



double iMPOImp::Compress(const SVCompressorR* compressor)
{
    int L=GetL();
    Vector<int> oldDws(L),newDws(L);
//    double truncationError=0.0;
    for (int ia=1;ia<=L;ia++)
    {
        oldDws(ia)=GetSiteOperator(ia)->GetDw12().Dw2;
        GetSiteOperator(ia)->CompressStd(DLeft ,compressor);
    }
    oldDws(L)=0;
    for (int ia=L;ia>=1;ia--)
    {
        GetSiteOperator(ia)->CompressStd(DRight,compressor);
        newDws(ia)=GetSiteOperator(ia)->GetDw12().Dw1;
    }
    newDws(1)=0;
    double percent=100-(100.0*Sum(newDws))/static_cast<double>(Sum(oldDws));
    cout << "% compression=" << std::fixed << std::setprecision(2) << percent << endl;
    return percent;
}


} //namespace

//---------------------------------------------------------------------------------
//
//  Make template instance.  For some reason we can's share the instance created in SiteOperatorImp.C
//
#define TYPE Matrix<double>
#include "oml/src/matrix.cpp"

