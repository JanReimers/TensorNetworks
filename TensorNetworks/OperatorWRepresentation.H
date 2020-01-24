#ifndef OPERATORWREPRESENTATION_H
#define OPERATORWREPRESENTATION_H

#include "TensorNetworks/Enums.H"
#include "TensorNetworks/Typedefs.H"

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

        virtual TensorNetworks::MatrixT GetW (TensorNetworks::Position,int m, int n) const=0;
        virtual TensorNetworks::ipairT  GetDw(TensorNetworks::Position) const=0;
};

#endif // OPERATORWREPRESENTATION_H
