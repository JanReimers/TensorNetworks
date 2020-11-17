#ifndef ITEBDSTATE_H
#define ITEBDSTATE_H

#include "TensorNetworks/Namespace.H"
#include <iosfwd>

namespace TensorNetworks
{

struct ONErrors
{
    double RightNormError;
    double Left_NormError;
    double RightOrthError;
    double Left_OrthError;
};

class iTEBDState
{
    public:
        iTEBDState() {};
        virtual ~iTEBDState() {};

        virtual void InitializeWith(State)=0;
        virtual void ReCenter(int isite)=0;
        virtual void IncreaseBondDimensions(int D)=0;
//--------------------------------------------------------------------------------------
//
//  Access to information
//
        virtual int GetL() const=0;
        virtual int Getp() const=0;
        virtual int GetD() const=0;
//--------------------------------------------------------------------------------------
//
//  Normalization members with and without SVD compression
//
        virtual void Canonicalize(Direction)=0;  //Vidals L-G-L-G-L canonical form
        virtual void Normalize(Direction LR)=0; //Use largest eigen value of transfer matrix
        virtual ONErrors Orthogonalize(SVCompressorC*)=0;

        virtual double GetExpectation (const MPO* o) const=0;
        virtual double GetExpectationmmnn (const Matrix4RT& Hlocal) const=0;
        virtual double GetExpectationmnmn (const Matrix4RT& expH  ) const=0;
        virtual void Report    (std::ostream&) const=0;
        virtual std::string GetNormStatus (         ) const=0;
        virtual void Apply(const Matrix4RT& expH,SVCompressorC*)=0;
};

}

#endif // ITEBDSTATE_H
