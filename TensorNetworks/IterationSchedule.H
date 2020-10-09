#ifndef ITERATIONSCHEDULE_H
#define ITERATIONSCHEDULE_H

#include "TensorNetworks/Epsilons.H"
#include "TensorNetworks/Enums.H"

#include <vector>
#include <iosfwd>

namespace TensorNetworks
{

struct IterationScheduleLine
{
    typedef  TrotterOrder  TOrder;
    //
    // Use these constructors for variational GS searches
    //
    IterationScheduleLine(int maxiter,int Dmax,const Epsilons&);
    IterationScheduleLine(int maxiter,int Dmax,int DeltaD,const Epsilons&);
    //
    // Use these constructors for iTime GS searches
    //
    IterationScheduleLine(int maxiter,int Dmax,double dt,TOrder,const Epsilons&);
    IterationScheduleLine(int maxiter,int Dmax,int maxOptIter,double dt,TOrder,const Epsilons&);

    int      itsMaxGSSweepIterations;  // Primary # of sweeps for variational GS search or i-time evolution
    int      itsMaxOptimizeIterations; // For post SVD compress optimization iterations
    int      itsDmax;                  // Max bond dimension
    int      itsDeltaD;                // Bond dimension increment
    double   itsdt;                    // Delta tau for i-time evolution
    TOrder   itsTrotterOrder;          // Order for descrete time steps/
    Epsilons itsEps;                   // Struct of various epsilons.

    friend std::ostream& operator<<(std::ostream&,const IterationScheduleLine&);
};

class IterationSchedule
{
        typedef  TrotterOrder  TOrder;
    public:
        IterationSchedule() : itsCurrentLine(0) {};
        void Insert(const IterationScheduleLine& l) {itsLines.push_back(l);}

        void begin() const {itsCurrentLine=0;}
        bool end  () const {return itsCurrentLine==itsLines.size();}
        const IterationScheduleLine& operator++(int) const //Move to next line, returns next line
        {
            itsCurrentLine++;
            return itsLines[itsCurrentLine];
        }
        const IterationScheduleLine& operator* () const {return itsLines[itsCurrentLine];}

        friend std::ostream& operator<<(std::ostream&,IterationSchedule&);
    private:
        mutable unsigned int itsCurrentLine;
        std::vector<IterationScheduleLine> itsLines;
};

}

#endif // ITERATIONSCHEDULE_H
