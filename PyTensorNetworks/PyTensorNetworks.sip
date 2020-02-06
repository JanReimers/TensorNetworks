// Define the SIP wrapper to the PyTensorNetworks library.

%Import PyPlotting.sip

%Module(name=PyTensorNetworks, language="C++")

%ModuleHeaderCode
#include "TensorNetworks/Enums.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/Operator.H"
#include "TensorNetworks/MatrixProductState.H"
#include "TensorNetworks/LRPSupervisor.H"
#include "TensorNetworks/Epsilons.H"
%End


%Include StdString.sip
//%Include StdVector.sip


%HideNamespace(name = TensorNetworks)

namespace TensorNetworks
{
    enum Position {Left,Bulk,Right};
    enum State {Product,Random,Neel};
};

class MultiPlotable
  : public Plotting::Plotable /Abstract/
{
public:
  virtual ~MultiPlotable();
  virtual void Show  (Plotting::Driver*,c_str title)=0;
  virtual void DrawOn(Plotting::Plotter*) const=0;
  virtual void Insert(Plotting::MultiGraph*)=0;
  virtual void Attach(Plotting::MultiGraph*)=0;
  virtual void Insert(const Plotting::Plotable*)=0;
};


class Operator /Abstract/
{
  public:
    Operator();
    virtual ~Operator();
};

class LRPSupervisor
{
    public:
        LRPSupervisor();
        virtual ~LRPSupervisor();

        virtual void ReadyToStart(c_str currentOperation);
        virtual void DoneOneStep(int level,c_str currentOperation,int site=-1);

};

struct Epsilons
{
    Epsilons();
    double itsEnergyConvergenceEpsilon;         //Keep sweeping until |dE|<eps
    double itsEigenConvergenceEpsilon;          //Used by eigen solver
    double itsEnergyVarienceConvergenceEpsilon; //increase D until <E^2>-<E>^2<eps
    double itsNormalizationEpsilon;             //Is normalized if max|A-I|<eps
    double itsSingularValueZeroEpsilon;         //s is deemed to be zero if s<eps
    double itsSparseMatrixEpsilon;              //matrix element A_ij is deemed to be zero if A_ij <eps
};

class MatrixProductState :public MultiPlotable /Abstract/
{
public:
    MatrixProductState();
    virtual ~MatrixProductState();
    virtual void InitializeWith(TensorNetworks::State)=0;

    virtual void   Normalize(TensorNetworks::Position LR,LRPSupervisor*)=0;
    virtual double FindGroundState(const Hamiltonian* H,int maxIter, const Epsilons& eps,LRPSupervisor*)=0;

    virtual double   GetExpectationIterate(const Operator* o) const=0;
    virtual double   GetExpectation       (const Operator* o1,const Operator* o2) const=0;

    virtual std::string GetNormStatus(int isite) const=0;
    virtual void Report(std::ostream&) const=0;
};

class OperatorWRepresentation /Abstract/;

class Hamiltonian : public Operator /Abstract/
{
    public:
        Hamiltonian();
        virtual ~Hamiltonian();
        virtual int GetL() const=0;
        virtual int Getp() const=0;
        //  Creates an MPS or operators with the correct lattice size and spin space.
        virtual MatrixProductState* CreateMPS     (int D,const Epsilons&) const=0;
        //virtual Operator*           CreateOperator(const OperatorWRepresentation*) const=0;
};



namespace TensorNetworks
{

    class Factory /Abstract/
    {
    public:
        Factory();
        virtual ~Factory();
        static const Factory* GetFactory() const;

        virtual Hamiltonian*             Make1D_NN_HeisenbergHamiltonian(int L, double S, double Jxy,double Jz, double hz) const=0 /Factory/;
        virtual Operator*                MakeOperator(const OperatorWRepresentation*, int L, int S2) const=0 /Factory/;
        virtual OperatorWRepresentation* MakeIdentityOperator() const=0 /Factory/;
    };

};



