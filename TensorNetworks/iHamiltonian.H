#ifndef iHAMILTONIAN_H
#define iHAMILTONIAN_H

#include "TensorNetworks/Namespace.H"
#include "TensorNetworks/iMPO.H"


namespace TensorNetworks
{
//
//  Abstract interface for a infinite lattice Hamiltonian represented as an iMPO.  Since the Hamiltonian
//  defines (knows) the unit cell extent (L) and the local Hilbert space (i.e. spin S) it is also a natural place
//  for construction of infinite wave function and other iMPO operator objects.
//

class iHamiltonian : public virtual iMPO
{
public:
    iHamiltonian() {}
    virtual ~iHamiltonian() {}

    virtual int    Getd() const=0; //Dimension of the one site Hilbert space
    virtual double GetS() const=0; //Spin of the one site Hilbert space
    //
    // Wave function zone:
    virtual iMPS*       CreateiMPS(int L,int D,double normEps=1e-12,double epsSV=1e-12) const=0;
    //    Creates an iTEBD state with the correct spin space. Second version allows different unit cell size
    virtual iTEBDState* CreateiTEBDState(      int D,iTEBDType,double normEps=1e-12,double epsSV=1e-12) const=0;
    virtual iTEBDState* CreateiTEBDState(int L,int D,iTEBDType,double normEps=1e-12,double epsSV=1e-12) const=0;
    //
    // Operator zone:
    //    Create full lattice of unit operators
    virtual iMPO* CreateiUnitOperator() const=0;
    virtual iMPO* CreateiH2Operator() const; //iMPO rep of Hamiltonian^2 for <E^2> calculations
    //  Create and exp(-dt*H) time evolution operator of a certain order in tau.
    virtual iMPO* CreateiMPO(double dt, TrotterOrder, CompressType,double epsMPO) const=0;
    //    Get a 2 site local matrix of order p^2 x p^2
    virtual Matrix4RT GetLocalMatrix(         ) const=0;
    virtual Matrix4RT GetExponentH  (double dt) const;
};

} //namespace

#endif //iHAMILTONIAN_H
