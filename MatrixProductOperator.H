#ifndef MATRIXPRODUCTOPERATOR_H
#define MATRIXPRODUCTOPERATOR_H

#include "TensorNetworks/MatrixProductState.H"
#include "TensorNetworks/MPOSite.H"

class MatrixProductOperator
{
    public:
        MatrixProductOperator(int L, int S2, int D);
        virtual ~MatrixProductOperator();

        int     GetL      () const {return itsL;}
        int     GetD      () const {return itsD;}
        int     Getp      () const {return itsp;}


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


};

#endif // MATRIXPRODUCTOPERATOR_H
