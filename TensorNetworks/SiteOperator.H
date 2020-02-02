#ifndef SITEOPERATOR_H
#define SITEOPERATOR_H

#include "TensorNetworks/Typedefs.H"

class Dw12;
//
//  The implementation of this class cache all the  <m|W|n> matrix elements
//  so it can used in hot loops.
//
class SiteOperator
{
public:
    SiteOperator() {};
    virtual ~SiteOperator() {};
    virtual TensorNetworks::ipairT GetDw  () const=0;
    virtual const Dw12*            GetDw12() const=0;
    //
    //  Generate MPO matrices.  m,n are quantum numbers for <m|Ha|n>
    //
    virtual const TensorNetworks::MatrixT GetW(int m, int n) const=0;
};

#endif // SITEOPERATOR_H
