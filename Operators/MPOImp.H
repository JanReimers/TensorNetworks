#ifndef MPOImp_H
#define MPOImp_H

#include "TensorNetworks/MPO.H"
#include "Containers/ptr_vector.h"

namespace TensorNetworks
{

//
//  All this class does is
//    1) store a string MPOs, each with left and right neighbour links.
//    2) serve up the site operators W(m,n) by site number
//  All operator are initialized as unit operators with Dw=1.
//  Non trivial MPOs are then created using the Combine function in MPO.H
//
class MPOImp
    : public virtual MPO
{
protected:
    typedef Matrix<MatrixRT> TensorT;

public:
    enum LoadWith {Identity,LoadLater};
    MPOImp(int L, double S,LoadWith);
    MPOImp(int L, double S,const TensorT& W);
    virtual ~MPOImp();

    virtual int    GetL() const {return itsL;}
    virtual int    Getd() const {return 2*itsS+1;}
    virtual double GetS() const {return itsS;}

    virtual const SiteOperator* GetSiteOperator(int isite) const;
    virtual       SiteOperator* GetSiteOperator(int isite);
protected:
    //
    //  Derived classes need to insert and then link the sites.
    //
    void    Insert(SiteOperator*);
    void    LinkSites(); //Set up left and right neighbour links
private:
    typedef optr_vector<SiteOperator*> SitesType;

    int       itsL;
    double    itsS;
    bool      areSitesLinked;
    SitesType itsSites;
};

}
#endif // MPO_H
