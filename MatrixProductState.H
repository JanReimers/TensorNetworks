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

        void SweepRight(const MatrixProductOperator* mpo);
        void SweepLeft (const MatrixProductOperator* mpo);

        double  GetOverlap(        ) const;
        MatrixT GetNeff   (int isite) const;

        // Hamiltion expectation
        Matrix6T GetEO(int isite, const MPOSite* mpos) const;
        Vector3T GetEOLeft_Iterate(const MatrixProductOperator* mpo,int isite) const;
        Vector3T GetEORightIterate(const MatrixProductOperator* mpo,int isite) const;
        Matrix6T GetHeffIterate   (const MatrixProductOperator* mpo,int isite) const;

        double ContractHeff(int isite,const Matrix6T& Heff) const;
        double ContractHeff(int isite,const MatrixT& Heff) const;
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
