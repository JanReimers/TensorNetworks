#ifndef MPOSITE_H
#define MPOSITE_H

#include "TensorNetworks/MatrixProductSite.H"
class Hamiltonian;

class MPOSite
{
    public:
        MPOSite(const Hamiltonian* H,int p, int D1, int D2);
        virtual ~MPOSite();

    protected:

    private:
        friend class MPOTesting;
        typedef MatrixProductSite::MatrixT MatrixT;
        typedef DMatrix<MatrixT> TensorT;

        int itsp; //2*S+1
        int itsD1,itsD2;

        TensorT itsWs;
};

#endif // MPOSITE_H
