#ifndef MPO_H
#define MPO_H

#include "TensorNetworks/Operator.H"
#include <iosfwd>

namespace TensorNetworks
{
//
//  MPO is an operator but it also has the combine and compress capabilities.
//  Most of the this interface can be implemented using a few abstract functions
//  from the Operator class.
//
class MPO : public virtual Operator
{
    public:
        MPO() {};
        virtual ~MPO() {};

        virtual int    GetMaxDw() const;
        virtual void   Combine(const Operator* O2);
        virtual double Compress(int Dmax, double minSV);
        virtual void   Report(std::ostream&) const;
};

}

#endif // MPO_H
