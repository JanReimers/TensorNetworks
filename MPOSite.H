#ifndef MPOSITE_H
#define MPOSITE_H

#include "TensorNetworks/MatrixProductSite.H"
#include <utility>

typedef std::pair<int,int> ipairT;


class Hamiltonian;

class MPOSite
{
        typedef MatrixProductSite::MatrixT MatrixT;
    public:
        MPOSite(const Hamiltonian* H,int p, int D1, int D2);
        virtual ~MPOSite();

        ipairT GetDw() const {return ipairT(itsDw1,itsDw2);}
        const MatrixT GetW(int m, int n) const {return itsWs(m+1,n+1);}

    protected:

    private:
        friend class MPOTesting;
        typedef DMatrix<MatrixT> TensorT;

        int itsp; //2*S+1
        int itsD1,itsD2;
        int itsDw1,itsDw2;

        TensorT itsWs;
};

#endif // MPOSITE_H
