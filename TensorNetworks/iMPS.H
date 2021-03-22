#ifndef IMATRIXPRODUCTSTATE_H
#define IMATRIXPRODUCTSTATE_H

#include "TensorNetworks/Namespace.H"
//#include <string>
//#include <iosfwd>


namespace TensorNetworks
{

class iMPS
{
public:
    virtual ~iMPS() {}
    virtual void InitializeWith(State)=0;
//--------------------------------------------------------------------------------------
//
//  Normalization members with and without SVD compression
//
    virtual void NormalizeSvD(Direction LR)=0;
    virtual void NormalizeQR (Direction LR)=0;
//--------------------------------------------------------------------------------------
//
// Find ground state.
//
    virtual double FindVariationalGroundState(const iHamiltonian* H,const IterationSchedule&)=0;
//--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//
    virtual double   GetExpectation (const iMPO* o) const=0; //This one checks the answer is real, the ..C version allows comples return values.
//    virtual dcmplx   GetExpectationC(const iMPO* o) const=0; //For non hermitian operators, S+, S-. Sy
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


} // namespace


#endif // IMATRIXPRODUCTSTATE_H
