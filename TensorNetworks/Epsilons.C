#include "TensorNetworks/Epsilons.H"

Epsilons::Epsilons(double default_eps)
    : itsEnergyConvergenceEpsilon        (default_eps)
    , itsEigenConvergenceEpsilon         (default_eps)
    //, itsSVDConvergenceEpsilon           (default_eps)
    , itsEnergyVarienceConvergenceEpsilon(default_eps)
    , itsNormalizationEpsilon            (default_eps)
    , itsSingularValueZeroEpsilon        (default_eps)
    , itsSparseMatrixEpsilon             (default_eps)
    , itsMaxIter                         (50)
{}
