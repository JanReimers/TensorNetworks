// Define the SIP wrapper to the PyTensorNetworks library.

%Module(name=PyTensorNetworks, language="C++")

%ModuleHeaderCode
#include "TensorNetworks/Enums.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/MPS.H"
#include "TensorNetworks/FullState.H"
#include "TensorNetworks/Epsilons.H"
#include "TensorNetworks/TNSLogger.H"
%End


%Include StdString.sip
//%Include StdVector.sip


%HideNamespace(name = TensorNetworks)

namespace TensorNetworks
{
    enum Position  {PLeft,PBulk,PRight};
    enum Direction {DLeft,DRight};
    enum State     {Product,Random,Neel};
    enum TrotterOrder {None,FirstOrder,SecondOrder,FourthOrder};

    typedef std::complex<double> dcmplx;
};

//class MultiPlotable
//  : public Plotting::Plotable /Abstract/
//{
//public:
//  virtual ~MultiPlotable();
//  virtual void Show  (Plotting::Driver*,c_str title)=0;
//  virtual void DrawOn(Plotting::Plotter*) const=0;
//  virtual void Insert(Plotting::MultiGraph*)=0;
//  virtual void Attach(Plotting::MultiGraph*)=0;
//  virtual void Insert(const Plotting::Plotable*)=0;
//  virtual void Select(int index)=0;
//};

namespace TensorNetworks
{

struct Epsilons
{
    Epsilons();
    double itsDelatEnergy1Epsilon;      // assume converged when <E_n> - <E_n-1> < eps
    double itsDelatEnergy2Epsilon;      // assume converged when <E^2> - <E>^2 < eps this tests convergence in D
    double itsDelatNormEpsilon;         // For Psi optimization iterations after SVD compression.
    double itsEigenSolverEpsilon;       // Decides convergence of whatever eigen solver is used.
    double itsNormalizationEpsilon;     // Is normalized if max|A-I|<eps
    double itsMPSCompressEpsilon;       // s is deemed to be zero if s<eps
    double itsMPOCompressEpsilon;       // s is deemed to be zero if s<eps
    double itsSparseMatrixEpsilon;      // matrix element A_ij is deemed to be zero if A_ij <eps
    double itsDeltaLambdaEpsilon;       // assume convergence when delta lambdaA or B < eps
};

struct IterationScheduleLine
{
    //
    // Use these constructors for variational GS searches
    //
    IterationScheduleLine(int maxiter,int Dmax,const Epsilons&);
//    IterationScheduleLine(int maxiter,int Dmax,int DeltaD,const Epsilons&);
    //
    // Use these constructors for iTime GS searches
    //
//    IterationScheduleLine(int maxiter,int Dmax,double dt,const Epsilons&);
//    IterationScheduleLine(int maxiter,int Dmax,int DeltaD,double dt,const Epsilons&);
//    IterationScheduleLine(int maxiter,int Dmax,double dt,TrotterOrder,const Epsilons&);
//    IterationScheduleLine(int maxiter,int Dmax,int DeltaD,double dt,TrotterOrder,const Epsilons&);
    IterationScheduleLine(int maxiter,int Dmax,int DeltaD,int maxOptIter,double dt,TrotterOrder,const Epsilons&);

    int      itsMaxGSSweepIterations;  // Primary # of sweeps for variational GS search or i-time evolution
    int      itsMaxOptimizeIterations; // For post SVD compress optimization iterations
    int      itsDmax;                  // Max bond dimension
    int      itsDeltaD;                // Bond dimension increment
    double   itsdt;                    // Delta tau for i-time evolution
    TensorNetworks::TrotterOrder   itsTrotterOrder;    // Order for descrete time steps/
    Epsilons itsEps;                   // Struct of various epsilons.

};


class IterationSchedule
{
    public:
        IterationSchedule();
        void Insert(const IterationScheduleLine& l);

};

class TNSLogger /Abstract/
{
    public:
        TNSLogger();
        virtual ~TNSLogger();
};

class MPO /Abstract/
{};

class MPS /Abstract/
{
public:
    MPS();
    virtual ~MPS();
    virtual void InitializeWith(TensorNetworks::State)=0;
    virtual void Freeze(int isite,double s)=0;

    virtual void   Normalize(TensorNetworks::Direction LR)=0;
    virtual void   NormalizeAndCompress(Direction LR,int Dmax,double eps);
//    virtual void   NormalizeAndCompress(Direction LR,SVCompressorC*)=0;

    virtual double FindVariationalGroundState(const Hamiltonian* H,const IterationSchedule&)=0;
    virtual double FindiTimeGroundState      (const Hamiltonian* H,const IterationSchedule&)=0;

    virtual int    Optimize(const MPS* Psi2,const IterationScheduleLine&)=0; // Minimize ||this-Psi2||^2

    virtual double   GetOverlap     (const MPS* Psi2) const=0; //<this|Psi2>
    virtual double   GetExpectation (const MPO* o) const=0; //This one checks the answer is real, the ..C version allows comples return values.
    //virtual dcmplx   GetExpectationC(const MPO* o) const=0; //For non hermitian operators, S+, S-. Sy

    virtual void  ApplyInPlace(const MPO* o) =0;
    virtual MPS*  Apply       (const MPO* o) const;

    virtual std::string GetNormStatus() const=0;
//    virtual void Report(std::ostream&) const=0;
};

class FullState /Abstract/
{
    public:
        FullState();
        virtual ~FullState();

        virtual int      GetSize     () const=0; //d^L
        virtual double   GetE        () const=0; //GS Eigen energy from power iterations
        // Iteracte |Psi'>=H*|Psi> until convergence to a ground state
        virtual double   PowerIterate   (const IterationScheduleLine&,const Hamiltonian&,bool quite=true)=0;
        // Use proper Lanczos to find the ground state
        virtual double   FindGroundState(const IterationScheduleLine&,const Hamiltonian&,bool quite=true)=0;

};


class Hamiltonian : public TensorNetworks::MPO /Abstract/
{
    public:
        Hamiltonian();
        virtual ~Hamiltonian();
        virtual int GetL() const=0;
        virtual int Getd() const=0;
        virtual MPS* CreateMPS(int D,double normEps=1e-12,double epsSV=1e-12) const=0/Factory/;
        // Create an exact (full) wave function abject suitable for this Hamiltonian
        virtual FullState*  CreateFullState () const=0/Factory/;

        virtual  MPO*       CreateH2Operator  () const;

};

};

namespace TensorNetworks
{

    class Factory /Abstract/
    {
    public:
        Factory();
        virtual ~Factory();
        static Factory* GetFactory();

        virtual Hamiltonian* Make1D_NN_HeisenbergHamiltonian(int L, double S, double Jxy,double Jz, double hz)=0 /Factory/;
        virtual TNSLogger*   MakeSPDLogger(int level)=0 /Factory/;
    };

};



