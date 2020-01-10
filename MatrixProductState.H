#ifndef MATRIXPRODUCTSTATE_H
#define MATRIXPRODUCTSTATE_H

#include "Misc/ptr_vector.h"
#include "TensorNetworks/MatrixProductSite.H"

class MatrixProductOperator;

class MatrixProductState
{
    public:
        typedef MatrixProductSite::MatrixT  MatrixT;
        typedef MatrixProductSite::Matrix6T Matrix6T;
        typedef MatrixProductSite::Vector3T Vector3T;
        typedef MatrixProductSite::VectorT  VectorT;
        typedef MatrixProductSite::VectorCT VectorCT;
        typedef MatrixProductSite::eType    eType;

        MatrixProductState(int L, int S2, int D);
        virtual ~MatrixProductState();

        void InitializeWith(MatrixProductSite::State);

        void Normalize(MatrixProductSite::Position LR);
        void Normalize(int isite); //Mixed Canonical normalization
        std::string GetNormStatus() const;
        void Report(std::ostream&) const;

        int     GetL      () const {return itsL;}
        int     GetD      () const {return itsD;}
        int     Getp      () const {return itsp;}
        ipairT GetDs(int isite) const {return itsSites[isite]->GetDs();}
//        Subscriptor  GetSuperMatrixSubscriptor (int isite) const;

        MatrixT GetMLeft (int isite) const;
        MatrixT GetMRight(int isite) const;
        MatrixT GetLeftNorm (int isite) const {return itsSites[isite]->GetLeftNorm();}
        MatrixT GetRightNorm(int isite) const {return itsSites[isite]->GetRightNorm();}
        bool    CheckNormalized(int isite,double eps) const;

        Matrix6T GetHeff       (const MatrixProductOperator *mpo,int isite) const;
        Matrix6T GetEOLeft     (const MatrixProductOperator *mpo,int isite) const;
        Matrix6T GetEORight    (const MatrixProductOperator *mpo,int isite) const;
        double   GetExpectation(const MatrixProductOperator *mpo) const;
        VectorCT Refine        (const MatrixProductOperator *mpo,int isite) const; //Minimize energy WRT the Ms in one site.

        void SweepRight(const MatrixProductOperator* mpo,bool quiet=true);
        void SweepLeft (const MatrixProductOperator* mpo,bool quiet=true);

        double  GetOverlap(        ) const;
        MatrixT GetNeff   (int isite) const;

        // Hamiltion expectation
        Matrix6T GetEO(int isite, const MPOSite* mpos) const;
        void     LoadHeffCaches   (const MatrixProductOperator* mpo);
        Vector3T GetEOLeft_Iterate(const MatrixProductOperator* mpo,int isite,bool cache=false) const;
        Vector3T GetEORightIterate(const MatrixProductOperator* mpo,int isite,bool cache=false) const;
        Matrix6T GetHeffIterate   (const MatrixProductOperator* mpo,int isite) const;
        double   GetExpectationIterate   (const MatrixProductOperator* mpo) const;

        double ContractHeff(int isite,const Matrix6T& Heff) const;
        double ContractHeff(int isite,const MatrixT& Heff) const;
        Vector3T GetHLeft_Cache(int isite) const;
        Vector3T GetHRightCache(int isite) const;
    protected:

    private:
        friend class MatrixProductTesting;
 //       friend class MPSNormTesting;

        typedef optr_vector<MatrixProductSite*> SitesType;
        typedef SitesType::      iterator  SIter;
        typedef SitesType::const_iterator cSIter;
        typedef SitesType::      reverse_iterator  rSIter;
        typedef SitesType::const_reverse_iterator crSIter;

        int itsL;   //Number of lattice sites
        int itsS2;  //2*S, avoid storing floats for S=n/2
        int itsD;   //Dimension of internal/virtual state space.
        int itsp;   //Number ot states for the spin on each site = 2S+1
        SitesType itsSites;
};

#endif // MATRIXPRODUCTSTATE_H
