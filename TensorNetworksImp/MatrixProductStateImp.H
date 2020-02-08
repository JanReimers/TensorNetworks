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
        typedef TensorNetworks::MatrixT  MatrixT;
        typedef TensorNetworks::VectorT  VectorT;
        typedef TensorNetworks:: ArrayT  ArrayT;
        typedef TensorNetworks::MatrixCT MatrixCT;
        typedef TensorNetworks::VectorCT VectorCT;
        typedef TensorNetworks::Matrix6T Matrix6T;
        typedef TensorNetworks::Vector3T Vector3T;
        typedef TensorNetworks::Vector4T Vector4T;
        typedef TensorNetworks::Matrix4T Matrix4T;
        typedef TensorNetworks::eType    eType;
//--------------------------------------------------------------------------------------
//
//  Conctruction and initialization zone
//
public:
        MatrixProductStateImp(int L, int S2, int D,const Epsilons& eps);
        virtual ~MatrixProductStateImp();

        virtual void InitializeWith(TensorNetworks::State);
//--------------------------------------------------------------------------------------
//
//  Access to information
//
        int     GetL        () const {return itsL;}
//        int     GetD        () const {return itsD;}
        int     Getp        () const {return itsp;}
        double  GetMaxDeltaE() const;

//--------------------------------------------------------------------------------------
//
//  Normalization members
//
        virtual void Normalize(TensorNetworks::Position LR,LRPSupervisor*);
        virtual void Normalize(int isite); //Mixed Canonical normalization
//--------------------------------------------------------------------------------------
//
// Find ground state
//
        virtual double   FindGroundState(const Hamiltonian* H,int maxIter, const Epsilons& eps,LRPSupervisor*);
//--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//

        virtual double   GetExpectationIterate(const Operator* o) const;
        virtual double   GetExpectation       (const Operator* o1,const Operator* o2) const;
//--------------------------------------------------------------------------------------
//
//    One site density matrix and expectation calculations
//
        virtual void CalculateOneSiteDMs(       LRPSupervisor*);
        virtual void CalculateTwoSiteDMs(int dx,LRPSupervisor*);
        virtual Matrix4T CalculateTwoSiteDM(int ia,int ib) const;
        virtual ArrayT GetOneSiteExpectation(const MatrixT &) const;
        virtual ArrayT GetOneSiteExpectation(const MatrixCT&) const;
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
//--------------------------------------------------------------------------------------
//
//  Allows unit test classes inside.
//
private:
        friend class MPOTesting;
        friend class MPSNormTesting;
        friend class MatrixProductTesting;
//--------------------------------------------------------------------------------------
//
//  For groound state search
//
        void     Refine        (const Hamiltonian *h,LRPSupervisor*,const Epsilons& eps,int isite); //Minimize energy WRT the Ms in one site.
        double   SweepRight    (const Hamiltonian* h,LRPSupervisor*,const Epsilons& eps,bool quiet=true);
        double   SweepLeft     (const Hamiltonian* h,LRPSupervisor*,const Epsilons& eps,bool quiet=true);
        void     UpdateBondData(int isite);

//--------------------------------------------------------------------------------------
//
// Calc Heff, calc L&R caches
//
        Matrix6T GetHeffIterate(const Hamiltonian* h,int isite) const;

        void     LoadHeffCaches(const Hamiltonian* h,LRPSupervisor*);
        Vector3T GetHLeft_Cache(int isite) const;
        Vector3T GetHRightCache(int isite) const;
        Vector3T GetEOLeft_Iterate(const Operator* o,LRPSupervisor*,int isite,bool cache=false) const;
        Vector3T GetEORightIterate(const Operator* o,LRPSupervisor*,int isite,bool cache=false) const;

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

        typedef std::vector<MatrixCT> MatrixListT;

//--------------------------------------------------------------------------------------
//
//  Data member
//

        int itsL;   //Number of lattice sites
        int itsS2;  //2*S, avoid storing floats for S=n/2
        //int itsD;   //Dimension of internal/virtual state space.
        int itsp;   //Number ot states for the spin on each site = 2S+1
        int itsNSweep;
        Epsilons    itsEpsilons;
        BondsType   itsBonds;
        SitesType   itsSites;
        MatrixListT itsOneSiteDMs; //One site densitymatrixes

        UniformMesh*  itsSitesMesh;
        Mesh*         itsBondsMesh;
        PlotableMesh* itsSitesPMesh;
        PlotableMesh* itsBondsPMesh;

        ArrayT itsSiteEnergies;  //Minimum eigen value of Heff on each site.
        ArrayT itsSiteEGaps;     //Gap to second eigen value on each site.
        ArrayT itsBondEntropies; // Sum(s^2*ln(s^2)) for each bond
        ArrayT itsBondMinSVs;    // smallest singluar value for each bond
        ArrayT itsBondRanks;    // Effective rank based on non zero singular values

        static GraphDefinition theGraphs[];
        static const int n_graphs;

};

#endif // MATRIXPRODUCTSTATEIMP_H
