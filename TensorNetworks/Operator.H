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
        virtual int   GetMaxDw() const;
        virtual void  Report(std::ostream&) const;
        virtual const SiteOperator* GetSiteOperator(int isite) const;
        virtual       SiteOperator* GetSiteOperator(int isite)      =0;
};

}

#endif // OPERATOR_H
