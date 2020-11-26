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
    virtual void ReCenter(int isite) const=0;
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
// Find ground state.  Return <E^2>-<E>^2
//
    virtual double FindiTimeGroundState      (const Hamiltonian* H,const IterationSchedule&)=0;
//--------------------------------------------------------------------------------------
//
//  iTime evolution
//
    virtual void   Apply     (const Matrix4RT& expH,SVCompressorC*)=0;
    virtual double ApplyOrtho(const Matrix4RT& expH,SVCompressorC*)=0;
    virtual double ApplyOrtho(const Matrix4RT& expH,SVCompressorC*,double eps,int maxIter)=0;

    virtual void   Apply     (const MPO* expH,SVCompressorC*)=0;
    virtual double ApplyOrtho(const MPO* expH,SVCompressorC*)=0;
    virtual double ApplyOrtho(const MPO* expH,SVCompressorC*,double eps,int maxIter)=0;
//--------------------------------------------------------------------------------------
//
//  Normalization, Conoization and orthogonalization.
//
    virtual void   Canonicalize (Direction)=0;  //Vidals L-G-L-G-L canonical form
    virtual void   Normalize    (Direction)=0; //Use largest eigen value of transfer matrix
    virtual double Orthogonalize(SVCompressorC*)=0;
//--------------------------------------------------------------------------------------
//
//  Contract expectation values.  Requires orthogonal state.
//  Hlocal and expH have different index ordering (for good reason) so they need
//  different implementations.
//
    virtual double GetExpectation (const MPO* o) const=0;
    virtual double GetExpectationmmnn (const Matrix4RT& Hlocal) const=0;
    virtual double GetExpectationmnmn (const Matrix4RT& expH  ) const=0;
//--------------------------------------------------------------------------------------
//
//  Reporting
//
    virtual void Report    (std::ostream&) const=0;
    virtual std::string GetNormStatus (         ) const=0;
};

}

#endif // ITEBDSTATE_H
