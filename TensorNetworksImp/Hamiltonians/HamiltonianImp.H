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
    , public virtual OperatorClient
    , private MPOImp
{
public:
    HamiltonianImp(int L, double S);
    virtual ~HamiltonianImp();
    //
    //  Hamiltonian interface
    //
    virtual int GetL () const {return MPOImp::GetL();}
    virtual int Getd () const {return 2*itsS+1;}
    virtual int GetMaxDw() const {return MPOImp::GetMaxDw();}

    virtual MPS*        CreateMPS          (int D          ,double normEps,double epsSV) const;
    virtual iTEBDState* CreateiTEBDState   (int D,iTEBDType,double normEps,double epsSV) const;
    virtual  MPO*       CreateUnitOperator () const;
    virtual iMPO*       CreateiUnitOperator() const;
    virtual  MPO*       CreateOperator     (double dt, TrotterOrder) const;
    virtual iMPO*       CreateiMPO         (double dt, TrotterOrder, double epsMPO) const;
    virtual iMPO*       CreateiMPO         () const; //iMPO rep of Hamiltonian
    virtual Matrix4RT   BuildLocalMatrix   () const;
    virtual FullState*  CreateFullState    () const;

protected:
//
//  OperatorClient interface
//
    virtual Dw12     GetDw12() const;
    virtual MatrixRT GetW   (int m, int n) const=0;
    virtual double GetH(int ma,int na,int mb,int nb,const SpinCalculator& sc) const=0;


    void   InitializeSites(); //Derived constructor must call this
    double I    (int m, int n) const;

    double itsS;
    Dw12 itsDw;
};

} //namespace

#endif // HAMILTONIANIMP_H
