#ifndef MATRIXPRODUCTSTATE_H
#define MATRIXPRODUCTSTATE_H

#include "TensorNetworks/Namespace.H"
#include "Containers/OneSiteDMs.H"
#include "Containers/TwoSiteDMs.H"
#include <string>
#include <iosfwd>


namespace TensorNetworks
{

class MPS
{
public:
    MPS() {}
    virtual ~MPS() {}
    virtual void InitializeWith(State)=0;
    virtual void Freeze(int isite,double s)=0;
    virtual MPS* Clone() const=0;
//--------------------------------------------------------------------------------------
//
//  Normalization members with and without SVD compression
//
    virtual void MixedCanonical(int isite)=0; //Mixed Canonical normalization

    virtual void Normalize(Direction LR);
    virtual void NormalizeAndCompress(Direction LR,int Dmax,double eps);
    virtual void NormalizeAndCompress(Direction LR,SVCompressorC*)=0;
//--------------------------------------------------------------------------------------
//
// Find ground state.  Return <E^2>-<E>^2
//
    virtual double FindVariationalGroundState(const Hamiltonian* H,const IterationSchedule&)=0;
    virtual double FindiTimeGroundState      (const Hamiltonian* H,const IterationSchedule&)=0;
//--------------------------------------------------------------------------------------
//
//  Vary this MPS to be as close as possible to Psi2 by minimizing ||this-Psi2||^2
//
    virtual int Optimize(const MPS* Psi2,const IterationScheduleLine&)=0; // Minimize ||this-Psi2||^2
//--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//
    virtual double   GetOverlap     (const MPS* Psi2) const=0; //<this|Psi2>
    virtual double   GetExpectation (const Operator* o) const=0; //This one checks the answer is real, the ..C version allows comples return values.
    virtual dcmplx   GetExpectationC(const Operator* o) const=0; //For non hermitian operators, S+, S-. Sy
//--------------------------------------------------------------------------------------
//
//  Operate on MPS with an MPO:  |Psi> -> O|Psi>.
//  In general this will increase the bond dimension from D to D*Dw
//  The overloaded operator is just syntatic suger, not sure how usefull it will be
//
    virtual void  ApplyInPlace(const Operator* o) =0;
    virtual MPS*  Apply       (const Operator* o) const;

//--------------------------------------------------------------------------------------
//
//    One and two site density matrix and expectation calculations
//
    virtual OneSiteDMs CalculateOneSiteDMs()=0;
    virtual TwoSiteDMs CalculateTwoSiteDMs()=0;

//--------------------------------------------------------------------------------------
//
//    Reporting
//
    virtual int         GetMaxD       (         ) const=0;
    virtual char        GetNormStatus (int isite) const=0;
    virtual std::string GetNormStatus (         ) const=0;
    virtual void        Report    (std::ostream&) const=0;
    virtual bool        IsRLNormalized(int isite) const=0; //  AAA..AMBB..BBB form at one site.
}; // class

inline MPS* operator*(const Operator& o, const MPS& psi)
{
    return psi.Apply(&o);
}

} // namespace


#endif // MATRIXPRODUCTSTATE_H
