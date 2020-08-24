#include "IterationSchedule.H"
#include <iostream>
#include <iomanip>

std::ostream& operator<<(std::ostream& os,const IterationScheduleLine& il)
{
    os.precision(3);
    os
    << std::setw(3) << std::fixed      << il.itsMaxGSSweepIterations       << " "
    << std::setw(3) << std::fixed      << il.itsMaxOptimizeIterations       << " "
    << std::setw(3) << std::fixed      << il.itsDmax                << " "
    << std::setw(6) << std::fixed      << il.itsdt                  << " "
    << il.itsEps;


    return os;
}

std::ostream& operator<<(std::ostream& os,IterationSchedule& is)
{
    os << "# #Sw #Opt  D  dTau  " << Epsilons::Header() << std::endl;
    for (is.begin();!is.end();is++)
        os << is.itsCurrentLine << " " << *is << std::endl;
    return os;
}
