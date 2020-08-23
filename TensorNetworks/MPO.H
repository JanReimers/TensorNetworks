#ifndef MPO_H
#define MPO_H

#include "TensorNetworks/Operator.H"

//
//  MPO is an operator but it can alse has the combine capability.
//
class MPO : public virtual Operator
{
    public:
        MPO() {};
        virtual ~MPO() {};

        virtual void Combine(const Operator* O2)=0;
};

#endif // MPO_H
