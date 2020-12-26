#include "TensorNetworksImp/iTEBD/iTEBDGates.H"
#include "TensorNetworks/Hamiltonian.H"



namespace TensorNetworks
{

iTEBDGates::iTEBDGates(int L,double S, int D,double normEps,double epsSV)
: iTEBDStateImp(L,S,D,normEps,epsSV)
{
    //ctor
}

iTEBDGates::~iTEBDGates()
{
    //dtor
}


void iTEBDGates::InitGates (const Hamiltonian* H,double dt,TrotterOrder to)
{
    itsGates.clear();
    Matrix4RT Hlocal=H->BuildLocalMatrix();
    switch (to)
    {
    case FirstOrder :
        itsGates.push_back(Hamiltonian::ExponentH(dt,Hlocal));
        itsGates.push_back(Hamiltonian::ExponentH(dt,Hlocal));
        break;
    case SecondOrder :
        itsGates.push_back(Hamiltonian::ExponentH(dt/2.0,Hlocal));
        itsGates.push_back(Hamiltonian::ExponentH(dt    ,Hlocal));
        itsGates.push_back(Hamiltonian::ExponentH(dt/2.0,Hlocal));
        break;
    case FourthOrder :
        {
            VectorRT ts(5);
            ts(1)=dt/(4-pow(4.0,1.0/3.0));
            ts(2)=ts(1);
            ts(3)=dt-2*ts(1)-2*ts(2);
            ts(4)=ts(2);
            ts(5)=ts(1);
            for (int it=1;it<=5;it++)
            {
                itsGates.push_back(Hamiltonian::ExponentH(ts(it)/2.0,Hlocal));
                itsGates.push_back(Hamiltonian::ExponentH(ts(it)    ,Hlocal));
                itsGates.push_back(Hamiltonian::ExponentH(ts(it)/2.0,Hlocal));
            }
            break;
        }
    default :
        assert(false);
    }
}

void iTEBDGates::Apply(SVCompressorC* comp, int center)
{
    iTEBDStateImp::Apply(itsGates,comp,center);
}

} //namespace
