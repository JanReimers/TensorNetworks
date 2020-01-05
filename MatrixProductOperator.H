#ifndef MATRIXPRODUCTOPERATOR_H
#define MATRIXPRODUCTOPERATOR_H

#include "TensorNetworks/MatrixProductState.H"
#include "TensorNetworks/MPOSite.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/Matrix6.H"


class MatrixProductOperator
{
    public:
       typedef MatrixProductSite::MatrixT  MatrixT;
       typedef MatrixProductSite::VectorT  VectorT;
       typedef MatrixProductSite::VectorCT VectorCT;
       typedef std::complex<double> eType;
       typedef Matrix6<eType> Matrix6T;
       typedef Matrix4<eType> Matrix4T;

        MatrixProductOperator(const Hamiltonian* H, int L, int S2, int D);
        virtual ~MatrixProductOperator();

        int     GetL      () const {return itsL;}
        int     GetD      () const {return itsD;}
        int     Getp      () const {return itsp;}

        Matrix6T GetHeff       (const MatrixProductState *mps,int isite) const;
        Matrix6T GetEOLeft     (const MatrixProductState *mps,int isite) const;
        Matrix6T GetEORight    (const MatrixProductState *mps,int isite) const;
        double   GetExpectation(const MatrixProductState *mps) const;
        VectorCT Refine        (      MatrixProductState *mps,int isite) const; //Minimize energy WRT the Ms in one site.
        void     SweepRight    (      MatrixProductState *mps) const;
        Hamiltonian::Position GetPosition(int isite) const;
    protected:

    private:
        friend class MPOTesting;

        typedef optr_vector<MPOSite*> SitesType;
        typedef SitesType::      iterator  SIter;
        typedef SitesType::const_iterator cSIter;

        int itsL;   //Number of lattice sites
        int itsD;   //Dimension of internal/virtual state space.
        int itsp;   //Number ot states for the spin on each site = 2S+1
        //
        // I think we only need to store 3 sites for OBC, left,bulk,right.
        //
        SitesType itsSites;
        const Hamiltonian* itsHamiltonian;
};

#endif // MATRIXPRODUCTOPERATOR_H
