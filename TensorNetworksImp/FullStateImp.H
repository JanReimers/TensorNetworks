#ifndef FULLSTATEIMP_H
#define FULLSTATEIMP_H

#include "TensorNetworks/FullState.H"
#include "TensorNetworksImp/Typedefs.H"

namespace TensorNetworks
{

//---------------------------------------------------------------------
//
//  Store a full state vector with p^L amplitues
//
class FullStateImp : public virtual FullState
{
    public:
        FullStateImp(int L, double S);
        virtual ~FullStateImp();

        virtual long int GetSize     () const {return itsN;}
        virtual double   GetE        () const {return itsE;}

        virtual double   PowerIterate   (const IterationScheduleLine&,const Hamiltonian&,bool quite=true);
        virtual double   FindGroundState(const IterationScheduleLine&,const Hamiltonian&,bool quite=true);
        virtual void     DoHContraction (int N, eType* xvec, eType* yvec, const Matrix4RT& Hlocal) const;

        virtual std::ostream& Dump(std::ostream&) const;

    protected:
        void    Normalize    (ArrayCT& amplitudes);
        double  Contract     (const Matrix4RT& Hlocal); // do H*Psi and fix up Psi
        ArrayCT Contract     (const Matrix4RT& Hlocal,const ArrayCT& oldAmpliudes) const; // just do H*Psi
        void    ContractLocal(int ia, const Matrix4RT& Hlocal, ArrayCT& newAmplitudes, const ArrayCT& oldAmpliudes) const;


    private:
        int      itsL;
        double   itsS;
        int      itsd;
        long int itsN; //p^L
        double   itsE; //Eigen energy from power iterations
        ArrayCT  itsAmplitudes; //2^L monster array
};

}
#endif // FULLSTATEIMP_H
