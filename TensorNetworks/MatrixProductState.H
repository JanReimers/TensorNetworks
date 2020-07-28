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

class MatrixProductState : public virtual MultiPlotable
{
    typedef TensorNetworks::MatrixT  MatrixT;
    typedef TensorNetworks::MatrixCT MatrixCT;
    typedef TensorNetworks::ArrayT   ArrayT;
    typedef TensorNetworks::eType    eType;
    public:
        MatrixProductState() {}
        virtual ~MatrixProductState() {}
        virtual void InitializeWith(TensorNetworks::State)=0;
        virtual void Freeze(int isite,double s)=0;
//--------------------------------------------------------------------------------------
//
//  Normalization members with and without SVD compression
//
        virtual void Normalize(TensorNetworks::Direction LR,LRPSupervisor*)=0;
        virtual void Normalize(int isite,LRPSupervisor*)=0; //Mixed Canonical normalization

        virtual void NormalizeAndCompress(TensorNetworks::Direction LR,int      Dmax,LRPSupervisor*)=0;
        virtual void NormalizeAndCompress(TensorNetworks::Direction LR,double epsMin,LRPSupervisor*)=0;
//--------------------------------------------------------------------------------------
//
// Find ground state.  Return <E^2>-<E>^2
//
        virtual double FindGroundState(const Hamiltonian* H,int maxIter, const Epsilons& eps,LRPSupervisor*)=0;
//--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//
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
        virtual MatrixProductState*  Apply       (const Operator* o) const=0;

        friend MatrixProductState* operator*(const Operator& o, const MatrixProductState& psi)
        {
            return psi.Apply(&o);
        }
//--------------------------------------------------------------------------------------
//
//    One and two site density matrix and expectation calculations
//
        virtual OneSiteDMs CalculateOneSiteDMs(LRPSupervisor*)=0;
        virtual TwoSiteDMs CalculateTwoSiteDMs(LRPSupervisor*)=0;

//--------------------------------------------------------------------------------------
//
//    Reporting
//
        virtual std::string GetNormStatus(int isite) const=0;
        virtual void Report(std::ostream&) const=0;
};

#endif // MATRIXPRODUCTSTATE_H
