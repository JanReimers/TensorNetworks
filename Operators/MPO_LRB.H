#ifndef MATRIXPRODUCTOPERATOR_H
#define MATRIXPRODUCTOPERATOR_H

#include "TensorNetworks/Namespace.H"
#include "TensorNetworks/Operator.H"
#include "Containers/ptr_vector.h"


namespace TensorNetworks
{
//
//  Represent an MPO with Left/Bulk/Right structure.  What this means is that we only need store
//  separate W matrices for the left, bulk and right locations.
//  This works for Hamiltonians
//
class MPO_LRB : public virtual Operator
{
    public:
        MPO_LRB(int L, double S);
        MPO_LRB(const OperatorWRepresentation* O, int L, double S);
        virtual ~MPO_LRB();

        virtual int   GetL() const {return itsL;}
        virtual const SiteOperator* GetSiteOperator(int isite) const
        {
            assert(itsSites.size()>0);
            return itsSites[GetPosition(isite)];
        }
    protected:
        void Init(const OperatorWRepresentation* O); //Derived classes must call this from constructor,

    private:
        Position GetPosition(int isite) const;
        friend class MPOTesting;

        typedef optr_vector<SiteOperator*> SitesType;
        typedef SitesType::      iterator  SIter;
        typedef SitesType::const_iterator cSIter;

        int itsL;   //Number of lattice sites
        int itsd;   //Number ot states for the spin on each site = 2S+1
        //
        // I think we only need to store 3 sites for OBC, left,bulk,right.
        //
        SitesType itsSites;
};

}

#endif // MATRIXPRODUCTOPERATOR_H
