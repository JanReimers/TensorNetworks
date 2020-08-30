#include "Operators/MPOImp.H"
#include "Operators/SiteOperatorImp.H"
#include "Operators/IdentityOperator.H"

MPOImp::MPOImp(int L, double S)
    : itsL(L)
    , itsp(2*S+1)
{
    assert(itsL>1);
    assert(itsp>1);
//
//  Load up the sites with unit operators
//
    OperatorWRepresentation* IdentityWOp=new IdentityOperator();
    itsSites.push_back(0); //Start count sites at index 1
    itsSites.push_back(new SiteOperatorImp(TensorNetworks::PLeft,IdentityWOp,itsp));
    for (int ia=2;ia<=itsL-1;ia++)
          itsSites.push_back(new SiteOperatorImp(TensorNetworks::PBulk,IdentityWOp,itsp));
    itsSites.push_back(new SiteOperatorImp(TensorNetworks::PRight,IdentityWOp,itsp));
//
//  Loop again and set neighbours.  Each site needs to know its neighbours in order to
//  carry out SVD tranfers, A[1]->U*s*VT, A=U, s*VT -> Transfered to next site.
//
    SiteOperatorImp* s=dynamic_cast<SiteOperatorImp*>(itsSites[1]);
    assert(s);
    s->SetNeighbours(0,itsSites[2]);
    for (int ia=2;ia<=itsL-1;ia++)
    {
        s=dynamic_cast<SiteOperatorImp*>(itsSites[ia]);
        assert(s);
        s->SetNeighbours(itsSites[ia-1],itsSites[ia+1]);
    }
    s=dynamic_cast<SiteOperatorImp*>(itsSites[itsL]);
    assert(s);
    s->SetNeighbours(itsSites[itsL-1],0);
}

MPOImp::~MPOImp()
{
    //dtor
}

void MPOImp::Combine(const Operator* O2)
{
    assert(itsL==O2->GetL());
    for (int ia=1;ia<=itsL;ia++)
    {
//        cout << "Site " << ia << " ";
        itsSites[ia]->Combine(O2->GetSiteOperator(ia));
    }
}

void MPOImp::Compress(int Dmax, double minSv)
{
    for (int ia=1;ia<itsL;ia++)
    {
//        cout << "Site " << ia << " ";
        itsSites[ia]->Compress(TensorNetworks::DLeft ,Dmax,minSv);
    }
//    Report(cout);
    for (int ia=itsL;ia>1;ia--)
        itsSites[ia]->Compress(TensorNetworks::DRight,Dmax,minSv);
}

void MPOImp::Report(std::ostream& os) const
{
    os << "Matrix Product Operator for " << itsL << " sites, p=" << itsp << endl;
    for (int ia=1;ia<=itsL;ia++)
    {
        os << "   Site " << ia << ": ";
        itsSites[ia]->Report(os);
        os << endl;
    }
}
