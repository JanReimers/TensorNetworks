#include "Operators/MPOImp.H"
#include "Operators/SiteOperatorImp.H"
#include "TensorNetworksImp/SVMPOCompressor.H"
#include "oml/vector_io.h"

namespace TensorNetworks
{

MPOImp::MPOImp(int L, double S)
    : itsL(L)
{
    int d=2*S+1;
    assert(itsL>1);
    assert(d>1);
//
//  Load up the sites with unit operators
//
    itsSites.push_back(0); //Start count sites at index 1
    itsSites.push_back(new SiteOperatorImp(d));
    for (int ia=2;ia<=itsL-1;ia++)
          itsSites.push_back(new SiteOperatorImp(d));
    itsSites.push_back(new SiteOperatorImp(d));
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



} //namespace
