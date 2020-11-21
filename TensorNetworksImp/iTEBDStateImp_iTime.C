#include "TensorNetworksImp/iTEBDStateImp.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/TNSLogger.H"

#include "TensorNetworksImp/Bond.H"

#include "NumericalMethods/ArpackEigenSolver.H"
#include "NumericalMethods/PrimeEigenSolver.H"
#include "NumericalMethods/LapackEigenSolver.H"
#include "NumericalMethods/LapackSVD.H"
#include "oml/cnumeric.h"
#include <iomanip>

namespace TensorNetworks
{


double iTEBDStateImp::FindiTimeGroundState(const Hamiltonian* H,const IterationSchedule& is)
{
    assert(Logger); //Make sure we have global logger.
    Canonicalize(TensorNetworks::DLeft);
    Logger->LogInfoV(0,"Initiate iTime GS iterations, D=%4d, Norm status=%s",GetMaxD(),GetNormStatus().c_str());
    MPO* H2=H->CreateH2Operator();
    H2->Report(cout);

    double E1=0;
    for (is.begin(); !is.end(); is++)
        E1=FindiTimeGroundState(H,H2,*is);

    double E2=GetExpectation(H2)/(itsL-1)/(itsL-1);
    E2= E2-E1*E1;
    delete H2;
    Logger->LogInfoV(0,"Finished iTime GS iterations D=%4d, <E>=%.9f, <E^2>-<E>^2=%.2e",GetMaxD(),E1,E2);
    return E2;
}

double iTEBDStateImp::FindiTimeGroundState(const Hamiltonian* H,const MPO* H2,const IterationScheduleLine& isl)
{
    assert(isl.itsDmax>0 || isl.itsEps.itsMPSCompressEpsilon>0);
    double dt=isl.itsdt;

    Matrix4RT Hlocal=H->BuildLocalMatrix();
    Matrix4RT expH=Hamiltonian::ExponentH(dt,Hlocal);
    int Dmax=GetMaxD();
    Logger->LogInfoV(1,"FindiTimeGroundState: begin iterations, dt=%.3f, epsE=%.2e D=%4d, isl.Dmax=%4d, Dw=%4d"
                     ,dt,isl.itsEps.itsDelatEnergy1Epsilon,Dmax,isl.itsDmax,H->GetMaxDw());
    SVCompressorC* mps_compressor =Factory::GetFactory()->MakeMPSCompressor(Dmax,0.0);
    int nOrthIter=100;
    OrthogonalizeI(mps_compressor,dt*dt/100,nOrthIter);
    ReCenter(1);
    double E1=GetExpectation(H)/(itsL-1);

    for (int D=Dmax;D<=isl.itsDmax;D+=isl.itsDeltaD)
    {
        if (D>Dmax)
        {
            IncreaseBondDimensions(D);
            delete mps_compressor;
            mps_compressor =Factory::GetFactory()->MakeMPSCompressor(D,0.0);
            Logger->LogInfoV(0,"Increasing D to %4d, isl.Dmax=%4d",D,isl.itsDmax);
            //
            //  Warm the state, i.e. get some real numbers in the matrix areas
            //
            Apply(expH,mps_compressor);
            ReCenter(2);
            Apply(expH,mps_compressor);
            ReCenter(1);
        }
        int niter=1;
        for (; niter<isl.itsMaxGSSweepIterations; niter++)
        {
            ApplyOrtho(expH,mps_compressor,dt*dt/100,nOrthIter);
            ReCenter(2);
            ApplyOrtho(expH,mps_compressor,dt*dt/100,nOrthIter);
            ReCenter(1);
            double Enew=GetExpectation(H)/(itsL-1);
            double dE=Enew-E1;
            E1=Enew;
            Logger->LogInfoV(3,"E=%.9f, dE=%.2e, niter=%4d",E1,dE,niter);
            if (fabs(dE)<=isl.itsEps.itsDelatEnergy1Epsilon) break;
        }
        OrthogonalizeI(mps_compressor,1e-10,nOrthIter);
        E1=GetExpectation(H);
        double E2=GetExpectation(H2);
        E2= E2-E1*E1;
        Logger->LogInfoV(1,"End %4d iterations, dt=%.3f,   E=%.9f, <dE^2>=%.9f, D=%4d, isl.Dmax=%4d"
                         ,niter,dt,E1,E2,D,isl.itsDmax);
    }
    delete mps_compressor;
    return E1;
}

void iTEBDStateImp::ApplyOrtho(const Matrix4RT& expH,SVCompressorC* comp,double eps,int maxIter)
{
    assert(comp);
    dVectorT theta=ContractTheta(expH);
    DiagonalMatrixRT lb=lambdaB();
    auto [gammap,lambdap]=OrthogonalizeI(theta,lb,eps,maxIter);
    UnpackOrthonormal(gammap,lambdap,comp);

}

//-----------------------------------------------------------------------------
//
//  THis follows PHYSICAL REVIEW B 78, 155117 2008 figure 14
//
void iTEBDStateImp::Apply(const Matrix4RT& expH,SVCompressorC* comp,bool orthogonalize)
{
    assert(comp);
    dVectorT theta=ContractTheta(expH);
    DiagonalMatrixRT lb=lambdaB();
    if (orthogonalize)
    {
        auto [gammap,lambdap]=Orthogonalize(theta,lb);
        UnpackOrthonormal(gammap,lambdap,comp);
    }
    else
    {
        UnpackOrthonormal(theta,lb,comp);
    }
}


}
