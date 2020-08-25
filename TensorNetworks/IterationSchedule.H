#ifndef ITERATIONSCHEDULE_H
#define ITERATIONSCHEDULE_H

#include "TensorNetworks/Epsilons.H"
#include "TensorNetworks/Enums.H"

#include <vector>
#include <iosfwd>

struct IterationScheduleLine
{
    typedef  TensorNetworks::TrotterOrder  TOrder;
    int      itsMaxGSSweepIterations;  // Primary # of sweeps for variational GS search or i-time evolution
    int      itsMaxOptimizeIterations; // For post SVD compress optimization iterations
    int      itsDmax;                  // Max bond dimension
    double   itsdt;                    // Delta tau for i-time evolution
    TOrder   itsTrotterOrder;          // Order for descrete time steps/
    Epsilons itsEps;                   // Struct of various epsilons.

    friend std::ostream& operator<<(std::ostream&,const IterationScheduleLine&);
};

class IterationSchedule
{
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

#endif // ITERATIONSCHEDULE_H
