#ifndef MATRIXPRODUCTSTATEIMP_H
#define MATRIXPRODUCTSTATEIMP_H

#include "TensorNetworks/MatrixProductState.H"
#include "TensorNetworks/Epsilons.H"
#include "TensorNetworksImp/MatrixProductSite.H"
#include "TensorNetworksImp/MultiPlotableImp.H"
#include "Functions/Mesh/TMesh.H"
#include "Misc/ptr_vector.h"

class PlotableMesh;

class MatrixProductStateImp
: public virtual MatrixProductState
, public virtual MultiPlotable
, private MultiPlotableImp
{
    public:
//--------------------------------------------------------------------------------------
//
//  Typedefs for various containers
//
private:
        typedef TensorNetworks::MatrixT   MatrixT;
        typedef TensorNetworks::VectorT   VectorT;
        typedef TensorNetworks:: ArrayT   ArrayT;
        typedef TensorNetworks::MatrixCT  MatrixCT;
        typedef TensorNetworks::VectorCT  VectorCT;
        typedef TensorNetworks::Matrix6CT Matrix6T;
        typedef TensorNetworks::Vector3CT Vector3T;
        typedef TensorNetworks::Vector4CT Vector4T;
        typedef TensorNetworks::Matrix4CT Matrix4T;
        typedef TensorNetworks::eType     eType;
//--------------------------------------------------------------------------------------
//
//  Conctruction and initialization zone
//
public:
        MatrixProductStateImp(int L, double S, int D,const Epsilons& eps);
        virtual ~MatrixProductStateImp();

        virtual void InitializeWith(TensorNetworks::State);
        virtual void Freeze(int isite,double s);
//--------------------------------------------------------------------------------------
//
//  Access to information
//
        int     GetL        () const {return itsL;}
        int     Getp        () const {return itsp;}
        double  GetMaxDeltaE() const;

//--------------------------------------------------------------------------------------
//
//  Normalization members with and without SVD compression
//
        virtual void Normalize(TensorNetworks::Direction LR,LRPSupervisor*);
        virtual void Normalize(int isite,LRPSupervisor*); //Mixed Canonical normalization

        virtual void NormalizeAndCompress(TensorNetworks::Direction LR,int      Dmax,LRPSupervisor*);
        virtual void NormalizeAndCompress(TensorNetworks::Direction LR,double epsMin,LRPSupervisor*);
//--------------------------------------------------------------------------------------
//
// Find ground state
//
        virtual double   FindGroundState(const Hamiltonian* H,int maxIter, const Epsilons& eps,LRPSupervisor*);
//--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//
        virtual double   GetOverlap     (const MatrixProductState* Psi2) const; //<this|Psi2>
        virtual double   GetExpectation (const Operator* o) const;
        virtual eType    GetExpectationC(const Operator* o) const;
        virtual double   GetExpectation (const Operator* o1,const Operator* o2) const;
//--------------------------------------------------------------------------------------
//
//  Operate on MPS with an MPO:  |Psi> -> O|Psi>.
//  In general this will increase the bond dimension from D to D*Dw
//
        virtual void                 ApplyInPlace(const Operator* o);
        virtual MatrixProductState*  Apply       (const Operator* o) const;
//--------------------------------------------------------------------------------------
//
//    One site density matrix and expectation calculations
//
        OneSiteDMs CalculateOneSiteDMs(LRPSupervisor*);
        TwoSiteDMs CalculateTwoSiteDMs(LRPSupervisor*);
//--------------------------------------------------------------------------------------
//
//    Reporting
//
        virtual std::string GetNormStatus(int isite) const;
        virtual void Report(std::ostream&) const;
//--------------------------------------------------------------------------------------
//
//      Multi Plotting interace
//
        virtual void MakeAllGraphs();
        virtual void Insert(Plotting::MultiGraph* graphs);
        virtual void Select(int index);
//--------------------------------------------------------------------------------------
//
//  Allows unit test classes inside.
//
private:
        friend class MPOTesting;
        friend class MPSNormTesting;
        friend class MatrixProductTesting;

        void InitPlotting();
//--------------------------------------------------------------------------------------
//
//  For groound state search
//
        void     Refine(TensorNetworks::Direction lr,const Hamiltonian *h,LRPSupervisor*,const Epsilons& eps,int isite); //Minimize energy WRT the Ms in one site.
        double   Sweep (TensorNetworks::Direction lr,const Hamiltonian* h,LRPSupervisor*,const Epsilons& eps);
        void     UpdateBondData(int isite);
        void     NormalizeSite(TensorNetworks::Direction lr,int isite,LRPSupervisor*);
        void     NormalizeAndCompressSite(TensorNetworks::Direction lr,int isite,int Dmax, double epsMin, LRPSupervisor*);

//--------------------------------------------------------------------------------------
//
// Calc Heff, calc L&R caches
//
        Matrix6T GetHeffIterate(const Hamiltonian* h,int isite) const;

        void     LoadHeffCaches(const Hamiltonian* h,LRPSupervisor*);
        Vector3T GetHeffCache(TensorNetworks::Direction lr,int isite) const;
        Vector3T GetEOLeft_Iterate(const Operator* o,LRPSupervisor*,int isite,bool cache=false) const;
        Vector3T GetEORightIterate(const Operator* o,LRPSupervisor*,int isite,bool cache=false) const;

        Matrix4T CalculateTwoSiteDM(int ia,int ib) const;
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


//--------------------------------------------------------------------------------------
//
//  Iterator Typedefs
//
        typedef optr_vector<MatrixProductSite*> SitesType;
        typedef optr_vector<Bond*>              BondsType;
        typedef SitesType::      iterator  SIter;
        typedef SitesType::const_iterator cSIter;
        typedef SitesType::      reverse_iterator  rSIter;
        typedef SitesType::const_reverse_iterator crSIter;

//--------------------------------------------------------------------------------------
//
//  Data members
//

        int    itsL;   //Number of lattice sites
        int    itsDmax; //Maximum bond dimension
        double itsS;  //2*S, avoid storing floats for S=n/2
        int    itsp;   //Number ot states for the spin on each site = 2S+1
        int    itsNSweep;
        int    itsSelectedSite; //Selected site for plots which show lots of data for one site.

        Epsilons    itsEpsilons;
        BondsType   itsBonds;
        SitesType   itsSites;

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

#endif // MATRIXPRODUCTSTATEIMP_H
