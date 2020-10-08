#ifndef ITEBDSTATE_H
#define ITEBDSTATE_H

#include "TensorNetworks/Enums.H"
#include "TensorNetworks/Typedefs.H"
#include "TensorNetworks/SVCompressor.H"
#include <iosfwd>


class iTEBDState
{
    protected:
        typedef TensorNetworks::Matrix4T Matrix4T;
    public:
        iTEBDState() {};
        virtual ~iTEBDState() {};

        virtual void InitializeWith(TensorNetworks::State)=0;
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
        virtual void Normalize(TensorNetworks::Direction LR);
        virtual void NormalizeAndCompress(TensorNetworks::Direction LR,SVCompressorC*)=0;

        virtual void Report    (std::ostream&) const=0;
        virtual void Apply(int isite,const Matrix4T& expH)=0;

    protected:

    private:
};

#endif // ITEBDSTATE_H
