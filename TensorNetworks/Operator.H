#ifndef OPERATOR_H
#define OPERATOR_H

#include "TensorNetworks/Namespace.H"

namespace TensorNetworks
{

class Operator
{
    public:

        Operator() {};
        virtual ~Operator() {};
        virtual int   GetL() const=0;
        virtual const SiteOperator* GetSiteOperator(int isite) const
        {
            return const_cast<Operator*>(this)->GetSiteOperator(isite);
        }
        virtual       SiteOperator* GetSiteOperator(int isite)      =0;
};

}

#endif // OPERATOR_H
