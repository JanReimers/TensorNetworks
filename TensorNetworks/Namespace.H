#ifndef NAMESPACE_H_INCLUDED
#define NAMESPACE_H_INCLUDED

#include "TensorNetworks/Enums.H"
#include "TensorNetworks/Typedefs.H"

namespace TensorNetworks
{
    class MPS;
    class MPO;
    class FullState;

    class Hamiltonian;

    class iTEBDState;
    class iHamiltonian;
    class iMPO;

    class SiteOperator;
    class SpinCalculator;
    class OperatorClient;

    template <class T> class SVCompressor;
    typedef SVCompressor<dcmplx> SVCompressorC;
    typedef SVCompressor<double> SVCompressorR;

    class Epsilons;
    class IterationSchedule;
    class IterationScheduleLine;

    class TNSLogger;
}

#endif // NAMESPACE_H_INCLUDED
