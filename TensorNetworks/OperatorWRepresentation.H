#ifndef OPERATORWREPRESENTATION_H
#define OPERATORWREPRESENTATION_H

#include "TensorNetworks/Namespace.H"

namespace TensorNetworks
{

//
//  The implementation of this class calculates the <m|W|n> matrix elements
//  directly so don't use this in hot loops.  Use SiteOperator instead, which caches
//  all the <m|W|n> matricies.
//
class OperatorWRepresentation
{
    public:
        typedef double ElementT;
        OperatorWRepresentation() {}
        virtual ~OperatorWRepresentation() {}

        virtual MatrixRT GetW   (Position,int m, int n) const=0;
        virtual Dw12     GetDw12(Position) const=0;
};

}
#endif // OPERATORWREPRESENTATION_H
