#ifndef MPO_H
#define MPO_H

#include "TensorNetworks/Operator.H"
#include <iosfwd>

//
//  MPO is an operator but it can alse has the combine capability.
//
class MPO : public virtual Operator
{
    public:
        MPO() {};
        virtual ~MPO() {};

        virtual void Combine(const Operator* O2)=0;
        virtual void Compress(int Dmax, double minSV)=0;
        virtual void Report(std::ostream&) const=0;
};

#endif // MPO_H
