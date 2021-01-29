#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "TensorNetworks/Namespace.H"
#include "TensorNetworks/MPO.H"


namespace TensorNetworks
{
//
//  Abstract interface for a finite lattice Hamiltonian represented as an MPO.  Since the Hamiltonian
//  defines (knows) the lattice extent (L) and the local Hilbert space (i.e. spin S) it is also a natural place
//  for construction of wave function and other operator objects.
//
class Hamiltonian : public virtual MPO
{
    public:
    Hamiltonian() {}
    virtual ~Hamiltonian() {}
    //virtual int GetL    () const=0; //# of lattice sites
    virtual int Getd    () const=0; //Dimension of the one site Hilbert space
    //
    // Wave function zone:
    //    Creates an MPS state with the correct lattice size and spin space.
    virtual MPS*       CreateMPS(int D,double normEps=1e-12,double epsSV=1e-12) const=0;
    //    Create an exact (full) wave function object suitable for this Hamiltonian
    virtual FullState* CreateFullState () const=0;
    //
    // Operator zone:
    //    Create full lattice of unit operators
    virtual  MPO* CreateUnitOperator() const=0;
    virtual  MPO* CreateH2Operator  () const;
    //    Create and exp(-dt*H) time evolution operator of a certain order in tau.
    virtual  MPO* CreateOperator(double dt, TrotterOrder) const=0;
    //    Get a 2 site local matrix of order p^2 x p^2
    virtual Matrix4RT GetLocalMatrix(         ) const=0;
    virtual Matrix4RT GetExponentH  (double dt) const;

    static  Matrix4RT GetExponentH  (double dt,const Matrix4RT& H12);
};

} //namespace

#endif //HAMILTONIAN_H
