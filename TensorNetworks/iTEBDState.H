#ifndef ITEBDSTATE_H
#define ITEBDSTATE_H

#include "TensorNetworks/Namespace.H"
#include <iosfwd>

namespace TensorNetworks
{

class iTEBDState
{
    public:
        iTEBDState() {};
        virtual ~iTEBDState() {};

        virtual void InitializeWith(State)=0;
 //--------------------------------------------------------------------------------------
//
//  Access to information
//
        virtual int GetL() const=0;
        virtual int Getp() const=0;
//--------------------------------------------------------------------------------------
//
//  Normalization members with and without SVD compression
//
        virtual void Normalize(Direction LR);
        virtual void NormalizeAndCompress(Direction LR,SVCompressorC*)=0;

        virtual void Report    (std::ostream&) const=0;
        virtual void Apply(int isite,const Matrix4RT& expH)=0;
};

}

#endif // ITEBDSTATE_H
