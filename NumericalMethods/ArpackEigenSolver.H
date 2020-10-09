#ifndef ARPACKEIGENSOLVER_H
#define ARPACKEIGENSOLVER_H

#include "TensorNetworks/Namespace.H"

class ArpackEigenSolver
{
    typedef TensorNetworks::MatrixCT MatrixCT;
    typedef TensorNetworks::VectorCT VectorCT;
public:
    ArpackEigenSolver();
    virtual ~ArpackEigenSolver();

    //Return value is: (EigenValues,EigenVectors)
    std::tuple<VectorCT,MatrixCT> Solve(const MatrixCT&, int NumEigenValues,const TensorNetworks::Epsilons& eps);


private:

};

#endif // ARPACKEIGENSOLVER_H
