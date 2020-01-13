#ifndef MATRIXPRODUCTSTATE_H
#define MATRIXPRODUCTSTATE_H

#include "Misc/ptr_vector.h"
#include "TensorNetworks/MatrixProductSite.H"

class MatrixProductOperator;

class MatrixProductState
{
    public:
//--------------------------------------------------------------------------------------
//
//  Typedefs for various containers
//
private:
        typedef MatrixProductSite::MatrixT  MatrixT;
        typedef MatrixProductSite::VectorT  VectorT;
        typedef MatrixProductSite::MatrixCT MatrixCT;
        typedef MatrixProductSite::VectorCT VectorCT;
        typedef MatrixProductSite::Matrix6T Matrix6T;
        typedef MatrixProductSite::Vector3T Vector3T;
        typedef MatrixProductSite::Vector4T Vector4T;
        typedef MatrixProductSite::eType    eType;
//--------------------------------------------------------------------------------------
//
//  Conctruction and initialization zone
//
public:
        MatrixProductState(int L, int S2, int D);
        virtual ~MatrixProductState();

        void InitializeWith(MatrixProductSite::State);
//--------------------------------------------------------------------------------------
//
//  Access to information
//
        int     GetL      () const {return itsL;}
        int     GetD      () const {return itsD;}
        int     Getp      () const {return itsp;}

//--------------------------------------------------------------------------------------
//
//  Norma;ization members
//
        void Normalize(MatrixProductSite::Position LR);
        void Normalize(int isite); //Mixed Canonical normalization
        bool CheckNormalized(int isite,double eps) const;
//--------------------------------------------------------------------------------------
//
// Find ground state
//
        int      FindGroundState(const MatrixProductOperator *mpo,int maxIter, double eps);
private:
        void     SweepRight    (const MatrixProductOperator* mpo,bool quiet=true);
        void     SweepLeft     (const MatrixProductOperator* mpo,bool quiet=true);
        void     Refine        (const MatrixProductOperator *mpo,int isite) const; //Minimize energy WRT the Ms in one site.
        void     LoadHeffCaches(const MatrixProductOperator* mpo);
        Vector3T GetHLeft_Cache(int isite) const;
        Vector3T GetHRightCache(int isite) const;
        double   GetMaxDeltaE() const;
        double   GetSigmaE   (const MatrixProductOperator* mpo) const;
        Matrix6T GetHeffIterate   (const MatrixProductOperator* mpo,int isite) const;

//--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//
public:

        double   GetOverlap           () const;
        double   GetExpectationIterate(const MatrixProductOperator* mpo) const;
        double   GetExpectation       (const MatrixProductOperator *mpo) const;
        double   GetExpectation       (const MatrixProductOperator* mpo1,const MatrixProductOperator* mpo2) const;
//--------------------------------------------------------------------------------------
//
//    Reporting
//
        std::string GetNormStatus() const;
        void Report(std::ostream&) const;

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
//  For low level ttesting only
//
private:

        MatrixCT GetMLeft (int isite) const;
        MatrixCT GetMRight(int isite) const;

        Matrix6T GetHeff       (const MatrixProductOperator *mpo,int isite) const;
        Matrix6T GetEOLeft     (const MatrixProductOperator *mpo,int isite) const;
        Matrix6T GetEORight    (const MatrixProductOperator *mpo,int isite) const;

        MatrixCT GetNeff   (int isite) const;

 //       Matrix6T GetEO(int isite, const MPOSite* mpos) const;
        Vector3T GetEOLeft_Iterate(const MatrixProductOperator* mpo,int isite,bool cache=false) const;
        Vector3T GetEORightIterate(const MatrixProductOperator* mpo,int isite,bool cache=false) const;

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
};

#endif // MATRIXPRODUCTSTATE_H
