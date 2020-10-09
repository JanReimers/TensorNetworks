#include "IterationSchedule.H"
#include <iostream>
#include <iomanip>
#include <cassert>

namespace TensorNetworks
{

IterationScheduleLine::IterationScheduleLine(int maxiter,int Dmax, const Epsilons& eps)
    : itsMaxGSSweepIterations(maxiter)
    , itsMaxOptimizeIterations(0) //not used for variational
    , itsDmax(Dmax)
    , itsDeltaD(1)
    , itsdt(0.0)  //not used for variational
    , itsTrotterOrder(None) //not used for variational
    , itsEps(eps)
{
    assert(itsMaxGSSweepIterations>0);
    assert(itsDmax>0);
}

IterationScheduleLine::IterationScheduleLine(int maxiter,int Dmax,int DD, const Epsilons& eps)
    : itsMaxGSSweepIterations(maxiter)
    , itsMaxOptimizeIterations(0) //not used for variational
    , itsDmax(Dmax)
    , itsDeltaD(DD)
    , itsdt(0.0)  //not used for variational
    , itsTrotterOrder(None) //not used for variational
    , itsEps(eps)
{
    assert(itsMaxGSSweepIterations>0);
    assert(itsDmax>0);
    assert(DD>0);
}

IterationScheduleLine::IterationScheduleLine(int maxiter,int Dmax,double dt,TrotterOrder o,const Epsilons& eps)
    : itsMaxGSSweepIterations(maxiter)
    , itsMaxOptimizeIterations(0)
    , itsDmax(Dmax)
    , itsDeltaD(1)
    , itsdt(dt)
    , itsTrotterOrder(o)
    , itsEps(eps)
{
    assert(itsMaxGSSweepIterations>0);
    assert(itsdt>0);
    assert(o!=None);
}

IterationScheduleLine::IterationScheduleLine(int maxiter,int Dmax,int maxOptIter,double dt,TrotterOrder o,const Epsilons& eps)
    : itsMaxGSSweepIterations(maxiter)
    , itsMaxOptimizeIterations(maxOptIter)
    , itsDmax(Dmax)
    , itsdt(dt)
    , itsTrotterOrder(o)
    , itsEps(eps)
{
    assert(itsMaxGSSweepIterations>0);
    assert(itsdt>0);
    assert(o!=None);
}


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

} //namespace
