#ifndef FULLSTATEIMP_H
#define FULLSTATEIMP_H

#include "TensorNetworks/FullState.H"
#include "TensorNetworksImp/Typedefs.H"
#include "NumericalMethods/SparseEigenSolver.H"
#include "oml/vector.h"

namespace TensorNetworks
{

//---------------------------------------------------------------------
//
//  Store a full state vector with p^L amplitues
//
template <class T> class FullStateImp
    : public virtual FullState
    , public virtual SparseEigenSolverClient<T>
{

    public:
        FullStateImp(int L, double S);
        virtual ~FullStateImp();

        virtual long int GetSize     () const {return itsN;}
        virtual double   GetE        () const {return itsE;}

        virtual double   PowerIterate   (const Hamiltonian&,double epsE,double epsPsi,int maxIter=1000);
        virtual double   FindGroundState(const Hamiltonian&,double epsE);
        virtual std::ostream& Dump(std::ostream&) const;
    protected:
//
// Implement PrimeEigenSolverClient interface
//
        virtual void  DoMatVecContraction (int N, const T* xvec, T* yvec) const;
        void          Normalize    (Vector<T>& amplitudes);
        double        OperateOverLattice     (); // do Psi=H*Psi and fix up Psi

        Vector<T> ContractOverLattice(const Vector<T>& oldAmpliudes) const; // just do H*Psi
        void      OperateLocal       (const Vector<T>& oldAmpliudes, Vector<T>& newAmplitudes,int ia ) const;


    private:
        int        itsL;
        double     itsS;
        int        itsd;
        long int   itsN; //p^L
        double     itsE; //Eigen energy from power iterations
        Vector<T>  itsAmplitudes; //2^L monster array
        Matrix4<T> itsHlocal; // Two site Ham matrix
};

}
#endif // FULLSTATEIMP_H
