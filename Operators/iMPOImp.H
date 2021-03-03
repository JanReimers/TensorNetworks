#ifndef iMPOImp_H
#define iMPOImp_H

#include "TensorNetworks/iMPO.H"
#include "Containers/ptr_vector.h"

namespace TensorNetworks
{
class OperatorBond;

class iMPOImp
    : public virtual iMPO
{
    typedef Matrix<MatrixRT> TensorT;
public:
    enum LoadWith {Identity};
    iMPOImp(int L, double S);
    iMPOImp(int L, double S,iMPOImp::LoadWith);
    iMPOImp(int L, const MatrixOR& W);
    iMPOImp(int L, const OperatorClient* W,MPOForm f);
    virtual ~iMPOImp();

    virtual int    GetL() const {return itsL;}
    virtual int    Getd() const {return 2*itsS+1;}
    virtual double GetS() const {return itsS;}

    virtual const  SiteOperator* GetSiteOperator(int isite) const;
    virtual        SiteOperator* GetSiteOperator(int isite);

    iMPO* MakeUnitcelliMPO(int unitcell) const;

protected:
    //
    //  Derived classes need to insert and then link the sites.
    //
    void Insert   (SiteOperator*);
    void LinkSites();//Set up left and right neighbour links

private:
    typedef optr_vector<SiteOperator*> SitesType;
    typedef optr_vector<OperatorBond*> BondsType;

    int       itsL;
    double    itsS;
    SitesType itsSites;
    BondsType itsBonds;
};

}
#endif // iMPO_H
