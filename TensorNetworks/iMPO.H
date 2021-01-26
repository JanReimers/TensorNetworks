#ifndef iMPO_H
#define iMPO_H

#include "TensorNetworks/Namespace.H"
#include <iosfwd>

namespace TensorNetworks
{
//
//  MPO is an operator but it also has the combine and compress capabilities.
//  Most of the this interface can be implemented using a few abstract functions
//  from the Operator class.
//
class iMPO
{
    public:
        iMPO() {};
        virtual ~iMPO() {};

        virtual iMPO* MakeUnitcelliMPO(int unitcell) const=0; //Contract horizontally to make iMPO for the whole unit cell.

        virtual const SiteOperator* GetSiteOperator(int isite) const;
        virtual       SiteOperator* GetSiteOperator(int isite)=0;

        virtual int   GetL() const=0;
        virtual int         GetMaxDw() const;
        virtual void        Report(std::ostream&) const; //Show Dws
        virtual void        Dump  (std::ostream&) const; //Show matrices
        virtual std::string GetNormStatus () const;
        virtual void   Combine(const iMPO* O2);
        virtual double CompressStd   (int Dmax, double epsSV);
        virtual double CompressStd   (const SVCompressorR*);
        virtual double CompressParker(int Dmax, double epsSV);
        virtual double CompressParker(const SVCompressorR*);
        virtual void   CanonicalForm(Direction LR);
};

}

#endif // iMPO_H
