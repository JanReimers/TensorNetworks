#ifndef MATRIXPRODUCTSTATE_H
#define MATRIXPRODUCTSTATE_H

#include "Misc/ptr_vector.h"
#include "TensorNetworks/MatrixProductSite.H"

class MatrixProductState
{
    public:
        typedef MatrixProductSite::MatrixT MatrixT;
        typedef MatrixProductSite::Matrix6T Matrix6T;
        typedef MatrixProductSite::VectorT VectorT;

        MatrixProductState(int L, int S2, int D);
        virtual ~MatrixProductState();

        void InitializeWithProductState();
        void InitializeWithRandomState();
        void Normalize(MatrixProductSite::Position LR);
        void Normalize(int isite); //Mixed Canonical normalization

        int     GetL      () const {return itsL;}
        int     GetD      () const {return itsD;}
        int     Getp      () const {return itsp;}
        ipairT GetDs(int isite) const {return itsSites[isite]->GetDs();}
//        Subscriptor  GetSuperMatrixSubscriptor (int isite) const;

        MatrixT GetMLeft (int isite) const;
        MatrixT GetMRight(int isite) const;
        MatrixT GetLeftNorm (int isite) const {return itsSites[isite]->GetLeftNorm();}
        MatrixT GetRightNorm(int isite) const {return itsSites[isite]->GetRightNorm();}

        double  GetOverlap(        ) const;
        MatrixT GetOverlap(int site) const;

        // Hamiltion expectation
        Matrix6T GetE(int isite, const MPOSite* mpos) const;
        double ConstractHeff(int isite,const Matrix6T& Heff) const;
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
