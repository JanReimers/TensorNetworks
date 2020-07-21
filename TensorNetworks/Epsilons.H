#ifndef EPSILONS_H
#define EPSILONS_H

//
//  Simple struct for passing many epsilons from python GUI to C++
//
struct Epsilons
{
    Epsilons(double default_eps=1e-12);
    double itsEnergyConvergenceEpsilon;         //Keep sweeping until |dE|<eps
    double itsEigenConvergenceEpsilon;          //Used by eigen solver
    //double itsSVDConvergenceEpsilon;            //Not currently used by SVD solver
    double itsEnergyVarienceConvergenceEpsilon; //increase D until <E^2>-<E>^2<eps
    double itsNormalizationEpsilon;             //Is normalized if max|A-I|<eps
    double itsSingularValueZeroEpsilon;         //s is deemed to be zero if s<eps
    double itsSparseMatrixEpsilon;              //matrix element A_ij is deemed to be zero if A_ij <eps
    int    itsMaxIter;                          //Max # of allowed iterations.
};

#endif // EPSILONS_H
