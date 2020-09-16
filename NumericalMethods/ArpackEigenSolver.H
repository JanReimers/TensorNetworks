#ifndef ARPACKEIGENSOLVER_H
#define ARPACKEIGENSOLVER_H

#include "TensorNetworksImp/Typedefs.H"

class Epsilons;

class ArpackEigenSolver
{
    typedef TensorNetworks::VectorT  VectorT;
    typedef TensorNetworks::VectorCT VectorCT;
    typedef TensorNetworks::MatrixCT MatrixCT;
public:
    ArpackEigenSolver();
    virtual ~ArpackEigenSolver();

    //Return value is: (EigenValues,EigenVectors)
    std::tuple<VectorCT,MatrixCT> Solve(const MatrixCT&, int NumEigenValues,const Epsilons& eps);


private:

};

#endif // ARPACKEIGENSOLVER_H
