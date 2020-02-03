#ifndef MATRIXPRODUCTSTATE_H
#define MATRIXPRODUCTSTATE_H

#include "TensorNetworks/Enums.H"
#include "TensorNetworks/MultiPlotable.H"
#include <string>
#include <vector>
#include <iostream>

class Operator;
class Hamiltonian;
class LRPSupervisor;
class Epsilons;

class MatrixProductState : public virtual MultiPlotable
{
    public:
        MatrixProductState() {}
        virtual ~MatrixProductState() {}
        virtual void InitializeWith(TensorNetworks::State)=0;
//--------------------------------------------------------------------------------------
//
//  Normalization members
//
        virtual void Normalize(TensorNetworks::Position LR,LRPSupervisor*)=0;
        virtual void Normalize(int isite)=0; //Mixed Canonical normalization
        virtual bool CheckNormalized(int isite,double eps) const=0;
//--------------------------------------------------------------------------------------
//
// Find ground state.  Return <E^2>-<E>^2
//
        virtual double FindGroundState(const Hamiltonian* H,int maxIter, const Epsilons& eps,LRPSupervisor*)=0;
//--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//
        virtual double   GetOverlap           () const=0;
        virtual double   GetExpectationIterate(const Operator* o) const=0;
        virtual double   GetExpectation       (const Operator *o) const=0;
        virtual double   GetExpectation       (const Operator* o1,const Operator* o2) const=0;
//--------------------------------------------------------------------------------------
//
//    Reporting
//
        virtual std::string GetNormStatus(int isite) const=0;
        virtual void Report(std::ostream&) const=0;
};

#endif // MATRIXPRODUCTSTATE_H
