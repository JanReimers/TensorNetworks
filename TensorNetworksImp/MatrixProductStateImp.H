#ifndef MATRIXPRODUCTSTATEIMP_H
#define MATRIXPRODUCTSTATEIMP_H

#include "TensorNetworks/MatrixProductState.H"
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
        typedef TensorNetworks::MatrixCT MatrixCT;
        typedef TensorNetworks::VectorCT VectorCT;
        typedef TensorNetworks::Matrix6T Matrix6T;
        typedef TensorNetworks::Vector3T Vector3T;
        typedef TensorNetworks::Vector4T Vector4T;
        typedef TensorNetworks::eType    eType;
//--------------------------------------------------------------------------------------
//
//  Conctruction and initialization zone
//
public:
        MatrixProductStateImp(int L, int S2, int D);
        virtual ~MatrixProductStateImp();

        virtual void InitializeWith(TensorNetworks::State);
//--------------------------------------------------------------------------------------
//
//  Access to information
//
        int     GetL      () const {return itsL;}
        int     GetD      () const {return itsD;}
        int     Getp      () const {return itsp;}

//--------------------------------------------------------------------------------------
//
//  Normalization members
//
        virtual void Normalize(TensorNetworks::Position LR,LRPSupervisor*);
        virtual void Normalize(int isite); //Mixed Canonical normalization
        virtual bool CheckNormalized(int isite,double eps) const;
//--------------------------------------------------------------------------------------
//
// Find ground state
//
        virtual int      FindGroundState(const Hamiltonian* H,int maxIter, double eps,LRPSupervisor*);
//--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//
public:

        virtual double   GetOverlap           () const;
        virtual double   GetExpectationIterate(const Operator* o) const;
        virtual double   GetExpectation       (const Operator *o) const;
        virtual double   GetExpectation       (const Operator* o1,const Operator* o2) const;
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
        void     SweepRight    (const Hamiltonian* h,LRPSupervisor*,bool quiet=true);
        void     SweepLeft     (const Hamiltonian* h,LRPSupervisor*,bool quiet=true);
        void     Refine        (const Hamiltonian *h,int isite) const; //Minimize energy WRT the Ms in one site.
        void     LoadHeffCaches(const Hamiltonian* h);
        Vector3T GetHLeft_Cache(int isite) const;
        Vector3T GetHRightCache(int isite) const;
        double   GetMaxDeltaE  () const;
        Matrix6T GetHeffIterate(const Hamiltonian* h,int isite) const;
        void     UpdateSiteData();

//--------------------------------------------------------------------------------------
//
//  For low level ttesting only
//
        MatrixCT GetMLeft (int isite) const;
        MatrixCT GetMRight(int isite) const;

        Matrix6T GetHeff       (const Hamiltonian* h,int isite) const;
        Matrix6T GetEOLeft     (const Operator* o   ,int isite) const;
        Matrix6T GetEORight    (const Operator* o   ,int isite) const;

        MatrixCT GetNeff   (int isite) const;

 //       Matrix6T GetEO(int isite, const MPOSite* mpos) const;
        Vector3T GetEOLeft_Iterate(const Operator* o,int isite,bool cache=false) const;
        Vector3T GetEORightIterate(const Operator* o,int isite,bool cache=false) const;

//--------------------------------------------------------------------------------------
//
//  Iterator Typedefs
//
        typedef optr_vector<MatrixProductSite*> SitesType;
        typedef SitesType::      iterator  SIter;
        typedef SitesType::const_iterator cSIter;
        typedef SitesType::      reverse_iterator  rSIter;
        typedef SitesType::const_reverse_iterator crSIter;

//--------------------------------------------------------------------------------------
//
//  Data member
//

        int itsL;   //Number of lattice sites
        int itsS2;  //2*S, avoid storing floats for S=n/2
        int itsD;   //Dimension of internal/virtual state space.
        int itsp;   //Number ot states for the spin on each site = 2S+1
        SitesType itsSites;

        UniformMesh*  itsSitesMesh;
        Mesh*  itsBondsMesh;
        PlotableMesh* itsSitesPMesh;
        PlotableMesh* itsBondsPMesh;

        Array<double> itsSiteEnergies;  //Minimum eigen value of Heff on each site.
        Array<double> itsSiteEGaps;     //Gap to second eigen value on each site.
        Array<double> itsBondEntropies; // Sum(s^2*ln(s^2)) for each bond
        Array<double> itsBondMinSVs;    // smallest singluar value for each bond

        static GraphDefinition theGraphs[];
        static const int n_graphs;

};

#endif // MATRIXPRODUCTSTATEIMP_H
