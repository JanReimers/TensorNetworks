#ifndef MATRIXPRODUCTSTATE_H
#define MATRIXPRODUCTSTATE_H

#include "TensorNetworks/Enums.H"
#include "TensorNetworks/Typedefs.H"
#include "TensorNetworks/MultiPlotable.H"
#include "Containers/OneSiteDMs.H"
#include "Containers/TwoSiteDMs.H"
#include <string>
#include <vector>
#include <iostream>

class Operator;
class Hamiltonian;
class LRPSupervisor;
class Epsilons;
class IterationSchedule;
class IterationScheduleLine;

class MPS : public virtual MultiPlotable
{
    typedef TensorNetworks::MatrixT  MatrixT;
    typedef TensorNetworks::MatrixCT MatrixCT;
    typedef TensorNetworks::ArrayT   ArrayT;
    typedef TensorNetworks::eType    eType;
    public:
        MPS() {}
        virtual ~MPS() {}
        virtual void InitializeWith(TensorNetworks::State)=0;
        virtual void Freeze(int isite,double s)=0;
        virtual MPS* Clone() const=0;
//--------------------------------------------------------------------------------------
//
//  Normalization members with and without SVD compression
//
        virtual void Normalize(TensorNetworks::Direction LR)=0;
        virtual void Normalize(int isite)=0; //Mixed Canonical normalization

        virtual void NormalizeAndCompress(TensorNetworks::Direction LR,int      Dmax)=0;
        virtual void NormalizeAndCompress(TensorNetworks::Direction LR,double epsMin)=0;
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
        virtual void Optimize(const MPS* Psi2,const IterationScheduleLine&)=0; // Minimize ||this-Psi2||^2
//--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//
        virtual double   GetOverlap     (const MPS* Psi2) const=0; //<this|Psi2>
        virtual double   GetExpectation (const Operator* o) const=0;
        virtual eType    GetExpectationC(const Operator* o) const=0; //For non hermitian operators, S+, S-. Sy
        virtual double   GetExpectation (const Operator* o1,const Operator* o2) const=0;
//--------------------------------------------------------------------------------------
//
//  Operate on MPS with an MPO:  |Psi> -> O|Psi>.
//  In general this will increase the bond dimension from D to D*Dw
//  The overloaded operator is just syntatic suger, not sure how usefull it will be
//
        virtual void                 ApplyInPlace(const Operator* o)      =0;
        virtual MPS*  Apply       (const Operator* o) const=0;

        friend MPS* operator*(const Operator& o, const MPS& psi)
        {
            return psi.Apply(&o);
        }
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
        virtual std::string GetNormStatus(int isite) const=0;
        virtual std::string GetNormStatus() const=0;
        virtual void Report(std::ostream&) const=0;
};

#endif // MATRIXPRODUCTSTATE_H
