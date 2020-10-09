#ifndef OPERATOR_H
#define OPERATOR_H

namespace TensorNetworks
{

class SiteOperator;
class Operator
{
    public:

        Operator() {};
        virtual ~Operator() {};
        virtual int   GetL() const=0;
        virtual const SiteOperator* GetSiteOperator(int isite) const=0;
};

}

#endif // OPERATOR_H
