#ifndef iMPO_H
#define iMPO_H

#include "TensorNetworks/MPO.H"

namespace TensorNetworks
{
//
//  iMPO is like an MPO but it infinitely repeats a unit cell (of width L).  For example
//  for L=2 we have iMPO = ..W[1]*W[2]*W[1]*W[2]...  In software the key distinction is
//  iMPO allow Dw>1 at the edges of unit cell.  This simple distinction make a whole new world
//  out of calculating expectation values
//
class iMPO : public virtual MPO
{
public:
    iMPO() {};
    virtual ~iMPO() {};

    virtual iMPO* MakeUnitcelliMPO(int unitcell) const=0; //Contract horizontally to make iMPO for the whole unit cell.
    virtual double Compress(CompressType ct,int Dmax, double epsSV) {return MPO::Compress(ct,Dmax,epsSV);}
    virtual double Compress(CompressType ct,const SVCompressorR*);
    virtual void   CanonicalForm();
    virtual void   CanonicalFormTri();
    virtual void   CanonicalFormQRIter(Direction lr);

};

}

#endif // iMPO_H
