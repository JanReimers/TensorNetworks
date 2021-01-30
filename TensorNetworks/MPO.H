#ifndef MPO_H
#define MPO_H

#include "TensorNetworks/Namespace.H"
#include <iosfwd>

namespace TensorNetworks
{
//
//  MPO is a finite lattice of matrix operators MPO = l*W[1]*W[2]*W[3]...*W[L-1]*W[L]*r
//  l and r are call boundary conditions which ensure no internal degrees of freedon (Dw=1)
//  at the edges of lattice.  Most MPO operators simply involve looping over the lattice.
//  Since we allow access to individual site operators (GetSiteOperator(i)) in this interface
//  we can implement most of the member function at this abstract level.  An example of OO
//  at its finest.
//
class MPO
{
public:
    MPO() {};
    virtual ~MPO() {};

    virtual const SiteOperator* GetSiteOperator(int isite) const=0;
    virtual       SiteOperator* GetSiteOperator(int isite)      =0;
    //
    //  Access to information
    //
    virtual int         GetL() const=0;
    virtual int         GetMaxDw() const;
    virtual void        Report(std::ostream&) const; //Show Dws, Fnorms, U/L status etc.
    virtual void        Dump  (std::ostream&) const; //Show all data in W matrices
    virtual std::string GetNormStatus () const;
    //
    //  High level MPO operations
    //
    virtual void   Product(const MPO* O2);
    virtual double Compress(CompressType,int Dmax, double epsSV);
    virtual double Compress(CompressType,const SVCompressorR*);
    virtual void   CanonicalForm(); //Sweep left and then right
};

}

#endif // MPO_H
