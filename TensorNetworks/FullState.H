#ifndef FULLSTATE_H
#define FULLSTATE_H

#include "TensorNetworks/Typedefs.H"
#include <iosfwd>

class Epsilons;
class Hamiltonian;

//--------------------------------------------------------------------------------
//
//  Abstract interface for full (exact) p^L wave function.
//
class FullState
{
    protected:
        typedef TensorNetworks::eType eType;
    public:
        FullState() {}
        virtual ~FullState() {}

        virtual long int GetSize     () const=0; //p^L
        virtual double   GetE        () const=0; //GS Eigen energy from power iterations
        // Iteracte |Psi'>=H*|Psi> until convergence to a ground state
        virtual double   PowerIterate   (const Epsilons&,const Hamiltonian&,bool quite=true)=0;
        // Use proper Lanczos to find the ground state
        virtual double   FindGroundState(const Epsilons&,const Hamiltonian&,bool quite=true)=0;
        // IO routines
        virtual std::ostream& Dump(std::ostream&) const=0;
        friend  std::ostream& operator<<(std::ostream& os,const FullState& psi)
        {
            return psi.Dump(os);
        }

};

#endif // FULLSTATE_H