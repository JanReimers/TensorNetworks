#ifndef MATRIXPRODUCTSTATEIMP_H
#define MATRIXPRODUCTSTATEIMP_H

#include "TensorNetworks/MPS.H"
#include "TensorNetworks/Epsilons.H"
#include "TensorNetworksImp/MPS/MPSSite.H"
#include "Containers/ptr_vector.h"
#include "Containers/rc_ptr.h"


// Required forward decs, for friend decs below
class MPOTests;
class MPSNormTesting;
class MPSTests;

namespace TensorNetworks
{

class MPSImp : public virtual MPS
{
//
//  Container typedefs
//
        typedef optr_vector<MPSSite*> SitesType;
        typedef optr_vector<Bond*>    BondsType;

//--------------------------------------------------------------------------------------
//
//  Construction and initialization zone
//
public:
        MPSImp(int L, double S, int D,double normEps,double epsSV);
        MPSImp(const MPSImp& mps);

        virtual ~MPSImp();

        virtual void InitializeWith(State);
        virtual void Freeze(int isite,double s);
        virtual MPS* Clone() const;
//--------------------------------------------------------------------------------------
//
//  Access to information
//
        virtual int     GetL        () const {return itsL;}
        virtual int     Getp        () const {return itsd;}
        virtual double  GetMaxDeltaE() const;
        virtual double  GetMaxDeltal() const; //Max delta lambda on bonds
        virtual double  GetMaxDeltal(const BondsType&) const; //Max delta lambda on bonds

//--------------------------------------------------------------------------------------
//
//  Normalization members with and without SVD compression
//
        virtual void MixedCanonical(int isite); //Mixed Canonical normalization
        virtual void CanonicalizeSite1(Direction LR, int isite,SVCompressorC*); //...lambda-A-lambda-A-lambda...
        virtual void CanonicalizeSite2(Direction LR, int isite,SVCompressorC*); //...lambda-Gamma-lambda-Gamma-lambda...

        virtual void NormalizeAndCompress(Direction LR,SVCompressorC*);
        virtual void NormalizeQR(Direction LR);
//--------------------------------------------------------------------------------------
//
// Find ground state
//
        virtual double FindVariationalGroundState(const Hamiltonian* H,const IterationSchedule&);
        virtual double FindiTimeGroundState      (const Hamiltonian* H,const IterationSchedule&);
//--------------------------------------------------------------------------------------
//
//  Vary this MPS to be as close as possible to Psi2 by minimizing ||this-Psi2||^2
//
        virtual int Optimize(const MPS* Psi2,const IterationScheduleLine&); // Minimize ||this-Psi2||^2
//--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//
        virtual double   GetOverlap     (const MPS* Psi2) const; //<this|Psi2>
        virtual double   GetExpectation (const MPO* o) const;
        virtual dcmplx   GetExpectationC(const MPO* o) const;
//--------------------------------------------------------------------------------------
//
//  Operate on MPS with an MPO:  |Psi> -> O|Psi>.
//  In general this will increase the bond dimension from D to D*Dw
//
        virtual void  ApplyInPlace(const MPO* o);
//--------------------------------------------------------------------------------------
//
//    One site density matrix and expectation calculations
//
        OneSiteDMs CalculateOneSiteDMs();
        TwoSiteDMs CalculateTwoSiteDMs();
//--------------------------------------------------------------------------------------
//
//    Reporting
//
        virtual int          GetMaxD      (         ) const;
        virtual char         GetNormStatus(int isite) const;
        virtual std::string  GetNormStatus(         ) const;
        virtual void            Report(std::ostream&) const;
        virtual bool        IsRLNormalized(int isite) const; //  AAA..AMBB..BBB form at one site.

protected:
    // Constructor used for iTEBD
        MPSImp(int L, double S, Direction lr,double normEps);
        void IncreaseBondDimensions(int D);
        void NormalizeAndCompressSite(Direction lr,int isite,SVCompressorC*);
        void NormalizeQRSite         (Direction lr,int isite);
//--------------------------------------------------------------------------------------
//
//  Loop Control
//
        int GetStart(Direction lr) const
        {
            return lr==DLeft ? 1 : itsL;
        }
        int GetStop(Direction lr) const
        { //These need to be one beyond, like stl iterators
            return lr==DLeft ? itsL+1 : 0;
        }
        int GetIncrement(Direction lr) const
        {
            return lr==DLeft ? 1 : -1;
        }

        Direction Invert(Direction lr)
        {
            return lr==DLeft ?  DRight : DLeft;
        }

//--------------------------------------------------------------------------------------
//
//  Allows unit test classes inside.
//
private:
        friend class ::MPOTests;
        friend class ::MPSNormTesting;
        friend class ::MPSTests;

        void InitSitesAndBonds(int D,double epsSV);

//--------------------------------------------------------------------------------------
//
//  For ground state search
//
        double FindVariationalGroundState(const Hamiltonian*,const IterationScheduleLine&);
        double FindiTimeGroundState      (const Hamiltonian*,const IterationScheduleLine&);
        void   Refine(Direction lr,const Hamiltonian*,const Epsilons&,int isite); //Minimize energy WRT the Ms in one site.
        void   Sweep (Direction lr,const Hamiltonian*,const Epsilons&);
        void   NormalizeSite             (Direction lr,int isite);
//-------------------------------------------------------------------------------------
//
//  hooks for derived classes to do live updates for UI interfaces or graphs
//
        virtual void   UpdateBondData(int isite) {};
        virtual void   UpdateEnergyData(int isite) {};
        virtual void   IterationDeltaE(int iter,double de){};
        virtual void   IterationEnergy(double E){};
//-------------------------------------------------------------------------------------
//
// Calc Heff, calc L&R caches
//
        Matrix6CT GetHeffIterate(const Hamiltonian* h,int isite) const;

        void     LoadHeffCaches(const Hamiltonian* h);
        Vector3CT GetHeffCache (Direction lr,int isite) const;
        Vector3CT CalcHeffLeft (const MPO* o,int isite,bool cache=false) const;
        Vector3CT CalcHeffRight(const MPO* o,int isite,bool cache=false) const;

        Matrix4CT CalculateTwoSiteDM(int ia,int ib) const;
//--------------------------------------------------------------------
//
//  Used for ||Psi1-Psi2||^2 optimizations.
//
        double   Sweep (Direction lr,const MPS* Psi2);
        void LoadCaches(const MPS* Psi2);
        MatrixCT GetRLCache(Direction lr,int isite) const;
        MatrixCT Calc12Left_Cache(const MPS* Psi2,int isite,bool cache=false) const;
        MatrixCT Calc12RightCache(const MPS* Psi2,int isite,bool cache=false) const;


//--------------------------------------------------------------------------------------
//
//  Iterator Typedefs
//
        typedef SitesType::      iterator  SIter;
        typedef SitesType::const_iterator cSIter;
        typedef SitesType::      reverse_iterator  rSIter;
        typedef SitesType::const_reverse_iterator crSIter;

//--------------------------------------------------------------------------------------
//
//  Data members
//
    protected:
        int    itsL;   //Number of lattice sites
        double itsS;  //2*S, avoid storing floats for S=n/2
        int    itsd;   //Number ot states for the spin on each site = 2S+1
        int    itsNSweep;
        double itsNormEps;

        BondsType  itsBonds;
        SitesType  itsSites;

}; // class

} //namespace
#define SiteLoop(ia) for (int ia=1;ia<=itsL;ia++)
#define ForLoop(LR) for (int ia=GetStart(LR);ia!=GetStop(LR);ia+=GetIncrement(LR))

#define CheckSiteNumber(ia)\
    assert(ia>=1);\
    assert(ia<=itsL);\

#define CheckBondNumber(ib)\
    assert(ib>=1);\
    assert(ib<itsL);\

#endif // MATRIXPRODUCTSTATEIMP_H
