#ifndef MPOCOMMON_H
#define MPOCOMMON_H

#include "TensorNetworks/iMPO.H"

namespace TensorNetworks
{
//
// Implement some MPO and iMPO functions that have the same implementation for both cases.
//
class MPO_Common
: public virtual iMPO
{
    public:
        MPO_Common() {};
        virtual ~MPO_Common() {};

//        virtual const SiteOperator* GetSiteOperator(int isite) const=0;
//        virtual       SiteOperator* GetSiteOperator(int isite)      =0;
//
//        virtual int         GetL() const=0;
//        virtual int         GetMaxDw() const;
//        virtual void        Report(std::ostream&) const; //Show Dws
//        virtual void        Dump  (std::ostream&) const; //Show matrices
//        virtual std::string GetNormStatus () const;
//
//
//        virtual double CompressStd   (int Dmax, double epsSV);
//        virtual double CompressStd   (const SVCompressorR*);
//        virtual double CompressParker(int Dmax, double epsSV);
//        virtual double CompressParker(const SVCompressorR*);
//        virtual void   CanonicalForm(Direction LR);
};

}

#endif // MPOCOMMON_H
