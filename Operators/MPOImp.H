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
//  Non trivial MPOs are then created using the Product function in MPO.H
//
class OperatorBond;


class MPOImp
    : public virtual MPO
{
protected:
    typedef Matrix<MatrixRT> TensorT;

public:
    enum LoadWith {Identity};
    MPOImp(int L, double S);
    MPOImp(int L, double S,LoadWith);
    MPOImp(int L, const OperatorClient* W,MPOForm f);
    virtual ~MPOImp();

    virtual int    GetL() const {return itsL;}
    virtual int    Getd() const {return 2*itsS+1;}
    virtual double GetS() const {return itsS;}
    virtual void   Report(std::ostream&) const; //Show Dws, Fnorms, U/L status etc.

    virtual const SiteOperator* GetSiteOperator(int isite) const;
    virtual       SiteOperator* GetSiteOperator(int isite);
protected:
    //
    //  Derived classes need to insert and then link the sites.
    //
    Position GetPosition(int L,int ia)
    {
        return ia==1 ? PLeft : (ia==L ? PRight : PBulk);
    }
    void    Insert(SiteOperator*);
    void    LinkSites(); //Set up left and right neighbour links
private:
    typedef optr_vector<SiteOperator*> SitesType;
    typedef optr_vector<OperatorBond*> BondsType;

    int       itsL;
    double    itsS;
    SitesType itsSites;
    BondsType itsBonds;
};

}
#endif // MPO_H
