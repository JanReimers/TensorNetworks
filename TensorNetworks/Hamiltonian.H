#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "TensorNetworks/Namespace.H"
#include "TensorNetworks/MPO.H"


namespace TensorNetworks
{

class Hamiltonian : public virtual MPO
{
    public:
        Hamiltonian() {}
        virtual ~Hamiltonian() {}
        virtual int GetL    () const=0; //# of lattice sites
        virtual int Getd    () const=0; //Dimension of the one site Hilbert space
        virtual int GetMaxDw() const=0;

        virtual const OperatorClient* GetClient() const=0;
        //  Creates an MPS or operators with the correct lattice size and spin space.
        virtual MPS*        CreateMPS       (int D,double normEps=1e-12,double epsSV=1e-12) const=0;
        virtual iTEBDState* CreateiTEBDState(int D,double normEps=1e-12,double epsSV=1e-12) const=0;
        // Create an exact (full) wave function abject suitable for this Hamiltonian
        virtual FullState*  CreateFullState () const=0;

        //  Create full lattice of unit operators
        virtual  MPO*       CreateUnitOperator() const=0;
        virtual iMPO*       CreateiUnitOperator() const=0;
        virtual  MPO*       CreateH2Operator  () const;
        //  Create and exp(-dt*H) time evolution operator of a certain order in tau.
        virtual  MPO*       CreateOperator(double dt, TrotterOrder) const=0;
        virtual iMPO*       CreateiMPO    (double dt, TrotterOrder, double epsMPO) const=0;
        // Get a 2 site local matrix of order p^2 x p^2
        virtual Matrix4RT   BuildLocalMatrix() const=0;

        static Matrix4RT ExponentH(double dt,const Matrix4RT& H12);
};

} //namespace

#endif //HAMILTONIAN_H
