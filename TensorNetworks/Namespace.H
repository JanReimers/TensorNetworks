#ifndef NAMESPACE_H_INCLUDED
#define NAMESPACE_H_INCLUDED

#include "TensorNetworks/Enums.H"
#include "TensorNetworks/Typedefs.H"

namespace TensorNetworks
{
    class MPS;
    class FullState;
    class iTEBDState;

    class Hamiltonian;
    class MPO;
    class Operator;
    class SiteOperator;
    class OperatorWRepresentation;
    class Dw12;

    template <class T> class SVCompressor;
    typedef SVCompressor<eType > SVCompressorC;
    typedef SVCompressor<double> SVCompressorR;

    class Epsilons;
    class IterationSchedule;
    class IterationScheduleLine;

    class TNSLogger;
}


#endif // NAMESPACE_H_INCLUDED
