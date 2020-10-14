#include "Operators/MPOImp.H"
#include "Operators/SiteOperatorImp.H"
#include "TensorNetworksImp/SVMPOCompressor.H"
#include "TensorNetworks/CheckSpin.H"
#include "oml/vector_io.h"

namespace TensorNetworks
{

MPOImp::MPOImp(int L, double S,LoadWith loadWith)
    : itsL(L)
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
            //
            Insert(new SiteOperatorImp(d));
            for (int ia=2;ia<=itsL-1;ia++)
                  Insert(new SiteOperatorImp(d));
            Insert(new SiteOperatorImp(d));
            break;
        }
    case LoadLater:
        {
            break;
        }
    }

}

MPOImp::~MPOImp()
{
    //dtor
}

void MPOImp::Insert(SiteOperator* so)
{
    assert(so);
    assert(static_cast<int>(itsSites.size())<=itsL);
    assert(!areSitesLinked);
    if (itsSites.size()==0) itsSites.push_back(0); //Dummy at index 0 so we start counting at index 1
    itsSites.push_back(so);
    if (static_cast<int>(itsSites.size())==itsL+1) LinkSites(); //Dummy at index 0 so we start counting at index 1
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
    areSitesLinked=true;
}


} //namespace
