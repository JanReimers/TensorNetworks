#ifndef MATRIXPRODUCTSTATEIMP_H
#define MATRIXPRODUCTSTATEIMP_H

#include "TensorNetworks/MPS.H"
#include "TensorNetworks/Epsilons.H"
#include "TensorNetworksImp/MPSSite.H"
#include "TensorNetworksImp/MultiPlotableImp.H"
#include "Functions/Mesh/TMesh.H"
#include "Misc/ptr_vector.h"

class PlotableMesh;
class IterationScheduleLine;

class MPSImp
: public virtual MPS
, public virtual MultiPlotable
, private MultiPlotableImp
{
    public:
//--------------------------------------------------------------------------------------
//
//  Typedefs for various containers
//
protected:
        typedef TensorNetworks::MatrixT   MatrixT;
        typedef TensorNetworks::VectorT   VectorT;
        typedef TensorNetworks:: ArrayT   ArrayT;
        typedef TensorNetworks::MatrixCT  MatrixCT;
        typedef TensorNetworks::VectorCT  VectorCT;
        typedef TensorNetworks::Matrix6CT Matrix6T;
        typedef TensorNetworks::Vector3CT Vector3CT;
        typedef TensorNetworks::Vector4CT Vector4CT;
        typedef TensorNetworks::Matrix4CT Matrix4CT;
        typedef TensorNetworks::eType     eType;
//--------------------------------------------------------------------------------------
//
//  Conctruction and initialization zone
//
public:
        MPSImp(int L, double S, int D,double normEps=1e-12,TNSLogger* s=0);
        MPSImp(const MPSImp& mps);

        virtual ~MPSImp();

        virtual void InitializeWith(TensorNetworks::State);
        virtual void Freeze(int isite,double s);
        virtual MPS* Clone() const;
//--------------------------------------------------------------------------------------
//
//  Access to information
//
        virtual int     GetL        () const {return itsL;}
        virtual int     Getp        () const {return itsd;}
        virtual double  GetMaxDeltaE() const;

//--------------------------------------------------------------------------------------
//
//  Normalization members with and without SVD compression
//
        virtual void Normalize(TensorNetworks::Direction LR);
        virtual void Normalize       (int isite); //Mixed Canonical normalization
        virtual void CanonicalizeSite(TensorNetworks::Direction LR, int isite); //...lambda-Gamma-lambda-Gamma-lambda...

        virtual void NormalizeAndCompress(TensorNetworks::Direction LR,int      Dmax);
        virtual void NormalizeAndCompress(TensorNetworks::Direction LR,double epsMin);
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
        virtual void Optimize(const MPS* Psi2,const IterationScheduleLine&); // Minimize ||this-Psi2||^2
//--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//
        virtual double   GetOverlap     (const MPS* Psi2) const; //<this|Psi2>
        virtual double   GetExpectation (const Operator* o) const;
        virtual eType    GetExpectationC(const Operator* o) const;
//--------------------------------------------------------------------------------------
//
//  Operate on MPS with an MPO:  |Psi> -> O|Psi>.
//  In general this will increase the bond dimension from D to D*Dw
//
        virtual void                 ApplyInPlace(const Operator* o);
        virtual MPS*  Apply       (const Operator* o) const;
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
        virtual char         GetNormStatus(int isite) const;
        virtual std::string  GetNormStatus(         ) const;
        virtual void            Report(std::ostream&) const;
        virtual bool        IsRLNormalized(int isite) const; //  AAA..AMBB..BBB form at one site.
//--------------------------------------------------------------------------------------
//
//      Multi Plotting interace
//
        virtual void MakeAllGraphs();
        virtual void Insert(Plotting::MultiGraph* graphs);
        virtual void Select(int index);


protected:
    // COnstructor used for iTEBD
        MPSImp(int L, double S, int D,TensorNetworks::Direction lr,double normEps=1e-12,TNSLogger* s=0);
//--------------------------------------------------------------------------------------
//
//  Loop Control
//
        int GetStart(TensorNetworks::Direction lr) const
        {
            return lr==TensorNetworks::DLeft ? 1 : itsL;
        }
        int GetStop(TensorNetworks::Direction lr) const
        { //These need to be one beyond, like stl iterators
            return lr==TensorNetworks::DLeft ? itsL+1 : 0;
        }
        int GetIncrement(TensorNetworks::Direction lr) const
        {
            return lr==TensorNetworks::DLeft ? 1 : -1;
        }

        TensorNetworks::Direction Invert(TensorNetworks::Direction lr)
        {
            return lr==TensorNetworks::DLeft ?  TensorNetworks::DRight : TensorNetworks::DLeft;
        }

//--------------------------------------------------------------------------------------
//
//  Allows unit test classes inside.
//
private:
        friend class MPOTesting;
        friend class MPSNormTesting;
        friend class MatrixProductTesting;

        void InitSitesAndBonds();
        void InitPlotting();
        static int GetD1(int a, int L, int d, int DMax);
        static int GetD2(int a, int L, int d, int DMax);
        static c_str SiteMessage(const std::string& message,int isite);
        void IncreaseBondDimensions(int D);

//--------------------------------------------------------------------------------------
//
//  For groound state search
//
        double FindVariationalGroundState(const Hamiltonian*,const IterationScheduleLine&);
        double FindiTimeGroundState      (const Hamiltonian*,const IterationScheduleLine&);
        void   Refine(TensorNetworks::Direction lr,const Hamiltonian*,const Epsilons&,int isite); //Minimize energy WRT the Ms in one site.
        void   Sweep (TensorNetworks::Direction lr,const Hamiltonian*,const Epsilons&);
        void   UpdateBondData(int isite);
        void   NormalizeSite             (TensorNetworks::Direction lr,int isite);
        void   NormalizeAndCompressSite  (TensorNetworks::Direction lr,int isite,int Dmax, double epsMin);
        void   SetCanonicalBondDimensions(TensorNetworks::Direction lr);
//-------------------------------------------------------------------------------------
//
// Calc Heff, calc L&R caches
//
        Matrix6T GetHeffIterate(const Hamiltonian* h,int isite) const;

        void     LoadHeffCaches(const Hamiltonian* h);
        Vector3CT GetHeffCache(TensorNetworks::Direction lr,int isite) const;
        Vector3CT CalcHeffLeft(const Operator* o,int isite,bool cache=false) const;
        Vector3CT CalcHeffRight(const Operator* o,int isite,bool cache=false) const;

        Matrix4CT CalculateTwoSiteDM(int ia,int ib) const;
//--------------------------------------------------------------------
//
//  Used for ||Psi1-Psi2||^2 optimizations.
//
        double   Sweep (TensorNetworks::Direction lr,const MPS* Psi2);
        void LoadCaches(const MPS* Psi2);
        MatrixCT GetRLCache(TensorNetworks::Direction lr,int isite) const;
        MatrixCT CalcHeffLeft(const MPS* Psi2,int isite,bool cache=false) const;
        MatrixCT CalcHeffRight(const MPS* Psi2,int isite,bool cache=false) const;


//--------------------------------------------------------------------------------------
//
//  Iterator Typedefs
//
        typedef optr_vector<MPSSite*> SitesType;
        typedef optr_vector<Bond*>    BondsType;
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
        int    itsDmax; //Maximum bond dimension
        double itsS;  //2*S, avoid storing floats for S=n/2
        int    itsd;   //Number ot states for the spin on each site = 2S+1
        int    itsNSweep;
        int    itsSelectedSite; //Selected site for plots which show lots of data for one site.

        double     itsNormEps;
        TNSLogger* itsLogger;
        BondsType  itsBonds;
        SitesType  itsSites;

        UniformMesh*  itsSitesMesh;
        Mesh*         itsBondsMesh;
        UniformMesh*  itsSVMesh;

        PlotableMesh* itsSitesPMesh;
        PlotableMesh* itsBondsPMesh;
        PlotableMesh* itsSVMeshPMesh;

        ArrayT itsSiteEnergies;  //Minimum eigen value of Heff on each site.
        ArrayT itsSiteEGaps;     //Gap to second eigen value on each site.
        ArrayT itsBondEntropies; // Sum(s^2*ln(s^2)) for each bond
        ArrayT itsBondMinSVs;    // smallest singluar value for each bond
        ArrayT itsBondRanks;     // Effective rank based on non zero singular values
        ArrayT itssSelectedEntropySpectrum; //full spectrum of SVs for a selected site.

        static GraphDefinition theGraphs[];
        static const int n_graphs;

};

#define SiteLoop(ia) for (int ia=1;ia<=itsL;ia++)
#define ForLoop(LR) for (int ia=GetStart(LR);ia!=GetStop(LR);ia+=GetIncrement(LR))

#define CheckSiteNumber(ia)\
    assert(ia>=1);\
    assert(ia<=itsL);\

#define CheckBondNumber(ib)\
    assert(ib>=1);\
    assert(ib<itsL);\

#endif // MATRIXPRODUCTSTATEIMP_H
