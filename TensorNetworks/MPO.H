#ifndef MPO_H
#define MPO_H

#include "TensorNetworks/Namespace.H"
#include <iosfwd>

namespace TensorNetworks
{
//
//  MPO is an operator but it also has the combine and compress capabilities.
//  Most of the this interface can be implemented using a few abstract functions
//  from the Operator class.
//
class MPO
{
    public:
        MPO() {};
        virtual ~MPO() {};
        virtual void ConvertToiMPO(int UnitCell)=0; //Strip away edge sites with w=1.

        virtual int   GetL() const=0;
        virtual int   GetMaxDw() const;
        virtual void  Report(std::ostream&) const; //Show Dws
        virtual void  Dump  (std::ostream&) const; //Show matrices
        virtual std::string GetNormStatus () const;
        virtual const SiteOperator* GetSiteOperator(int isite) const;
        virtual       SiteOperator* GetSiteOperator(int isite)      =0;

        virtual void   Combine(const MPO* O2);
        virtual void   Combine(const MPO* O2,double factor);
        virtual double CompressStd   (int Dmax, double epsSV);
        virtual double CompressStd   (const SVCompressorR*);
        virtual double CompressParker(int Dmax, double epsSV);
        virtual double CompressParker(const SVCompressorR*);
        virtual void   CanonicalForm(Direction LR);
};

}

#endif // MPO_H
