#ifndef HAMILTONIANIMP_H
#define HAMILTONIANIMP_H

#include "TensorNetworks/Hamiltonian.H"
#include "Operators/SiteOperatorImp.H"
#include "Operators/MPOImp.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworksImp/SpinCalculator.H"

namespace TensorNetworks
{
//
//  Common imp for all Hamiltonians
//
class HamiltonianImp
    : public virtual Hamiltonian
    , private MPOImp
{
public:
    HamiltonianImp(int L, const OperatorClient* W);
    virtual ~HamiltonianImp();
    //
    //  MPO interface
    //
//    virtual const SiteOperator* GetSiteOperator(int isite) const {return MPOImp::GetSiteOperator(isite);}
//    virtual       SiteOperator* GetSiteOperator(int isite)       {return MPOImp::GetSiteOperator(isite);}

    //
    //  Hamiltonian interface
    //
    virtual int GetL () const {return MPOImp::GetL();}
    virtual int Getd () const {return 2*itsS+1;}
//    virtual int GetMaxDw() const {return MPOImp::GetMaxDw();}

    virtual MPS*       CreateMPS          (int D          ,double normEps,double epsSV) const;
    virtual MPO*       CreateUnitOperator () const;
    virtual MPO*       CreateOperator     (double dt, TrotterOrder) const;
    virtual FullState* CreateFullState    () const;
    virtual Matrix4RT  GetLocalMatrix(         ) const {return itsH12;}

protected:
//    double itsS;
    Dw12 itsDw;
    Matrix4RT itsH12;
};

} //namespace

#endif // HAMILTONIANIMP_H
