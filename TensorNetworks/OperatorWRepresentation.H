#ifndef OPERATORWREPRESENTATION_H
#define OPERATORWREPRESENTATION_H

#include "TensorNetworks/Enums.H"
#include "TensorNetworks/Typedefs.H"

class Dw12;
//
//  The implementation of this class calculates the <m|W|n> matrix elements
//  directly so don't use this in hot loops.  Use SiteOperator instead, which caches
//  all the <m|W|n> matricies.
//
class OperatorWRepresentation
{
    public:
        typedef double ElementT;
        typedef TensorNetworks::MatrixRT MatrixRT;
        OperatorWRepresentation() {}
        virtual ~OperatorWRepresentation() {}

        virtual MatrixRT GetW   (TensorNetworks::Position,int m, int n) const=0;
        virtual Dw12    GetDw12(TensorNetworks::Position) const=0;
};

#endif // OPERATORWREPRESENTATION_H
