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
