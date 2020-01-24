#ifndef MATRIXPRODUCTOPERATOR_H
#define MATRIXPRODUCTOPERATOR_H

#include "TensorNetworks/Operator.H"
#include "TensorNetworks/Enums.H"
#include "Misc/ptr_vector.h"

class SiteOperator;
class OperatorWRepresentation;

class MatrixProductOperator : public virtual Operator
{
    public:
        MatrixProductOperator(int L, int S2);
        MatrixProductOperator(const OperatorWRepresentation* O, int L, int S2);
        virtual ~MatrixProductOperator();

        const SiteOperator* GetSiteOperator(int isite) const
        {
            assert(itsSites.size()>0);
            return itsSites[GetPosition(isite)];
        }
    protected:
        void Init(const OperatorWRepresentation* O); //Derived classes must call this from constructor,

    private:
        TensorNetworks::Position GetPosition(int isite) const;
        friend class MPOTesting;

        typedef optr_vector<SiteOperator*> SitesType;
        typedef SitesType::      iterator  SIter;
        typedef SitesType::const_iterator cSIter;

        int itsL;   //Number of lattice sites
        int itsp;   //Number ot states for the spin on each site = 2S+1
        //
        // I think we only need to store 3 sites for OBC, left,bulk,right.
        //
        SitesType itsSites;
};

#endif // MATRIXPRODUCTOPERATOR_H
