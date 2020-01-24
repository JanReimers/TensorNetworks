#ifndef OPERATOR_H
#define OPERATOR_H

class SiteOperator;
class Operator
{
    public:

        Operator() {};
        virtual ~Operator() {};
        virtual const SiteOperator* GetSiteOperator(int isite) const=0;
};

#endif // OPERATOR_H
