#ifndef FULLSTATE_H
#define FULLSTATE_H

#include "TensorNetworks/Namespace.H"
#include <iosfwd>


namespace TensorNetworks
{

//--------------------------------------------------------------------------------
//
//  Abstract interface for full (exact) p^L wave function.
//
class FullState
{
    public:
        FullState() {}
        virtual ~FullState() {}

        virtual long int GetSize     () const=0; //p^L
        virtual double   GetE        () const=0; //GS Eigen energy from power iterations
        // Iterate |Psi'>=H*|Psi> until convergence to a ground state
        virtual double   PowerIterate   (const Hamiltonian&,double epsE,double epsPsi,int maxIter=1000)=0;
        // Use proper Lanczos to find the ground state in far less iterations
        virtual double   FindGroundState(const Hamiltonian&,double epsE)=0;
        // IO routines
        virtual std::ostream& Dump(std::ostream&) const=0;
        friend  std::ostream& operator<<(std::ostream& os,const FullState& psi)
        {
            return psi.Dump(os);
        }

};

}
#endif // FULLSTATE_H
