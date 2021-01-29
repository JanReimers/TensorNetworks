#ifndef iHAMILTONIAN_H
#define iHAMILTONIAN_H

#include "TensorNetworks/Namespace.H"
#include "TensorNetworks/iMPO.H"


namespace TensorNetworks
{

class iHamiltonian : public virtual iMPO
{
    public:
        iHamiltonian() {}
        virtual ~iHamiltonian() {}
//        virtual int GetL    () const=0; //# of lattice sites
        virtual int Getd    () const=0; //Dimension of the one site Hilbert space
//        virtual int GetMaxDw() const=0;

        //  Creates an MPS or operators with the correct lattice size and spin space.
        virtual iTEBDState* CreateiTEBDState(int D,iTEBDType,double normEps=1e-12,double epsSV=1e-12) const=0;

        //  Create full lattice of unit operators
        virtual iMPO*       CreateiUnitOperator() const=0;
        virtual iMPO*       CreateiH2Operator() const; //iMPO rep of Hamiltonian^2 for <E^2> calculations
        //  Create and exp(-dt*H) time evolution operator of a certain order in tau.
        virtual iMPO*       CreateiMPO    (double dt, TrotterOrder, double epsMPO) const=0;

        virtual Matrix4RT GetLocalMatrix(         ) const=0;
        virtual Matrix4RT GetExponentH  (double dt) const;
};

} //namespace

#endif //iHAMILTONIAN_H
