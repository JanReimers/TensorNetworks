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

    double E1=0;
    for (is.begin(); !is.end(); is++)
        E1=FindiTimeGroundState(H,*is);

    MPO* H2=H->CreateH2Operator();
    double E2=GetExpectation(H2);
    E2= E2-E1*E1;
    delete H2;
    Logger->LogInfoV(0,"Finished iTime GS iterations D=%4d, <E>=%.9f, <E^2>-<E>^2=%.2e",GetMaxD(),E1,E2);
    return E2;
}

double iTEBDStateImp::FindiTimeGroundState(const Hamiltonian* H,const IterationScheduleLine& isl)
{
    assert(isl.itsDmax>0 || isl.itsEps.itsMPSCompressEpsilon>0);


    Matrix4RT Hlocal=H->BuildLocalMatrix();
    Matrix4RT expH=Hamiltonian::ExponentH(isl.itsdt,Hlocal);
    int Dmax=GetMaxD();
    Logger->LogInfoV(1,"FindiTimeGroundState: begin iterations, dt=%.3f, epsE=%.2e D=%4d, isl.Dmax=%4d, Dw=%4d"
                     ,isl.itsdt,isl.itsEps.itsDelatEnergy1Epsilon,Dmax,isl.itsDmax,H->GetMaxDw());
    SVCompressorC* mps_compressor =Factory::GetFactory()->MakeMPSCompressor(Dmax,0.0);
    Orthogonalize(mps_compressor);
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
        }
        int niter=1;
        for (; niter<isl.itsMaxGSSweepIterations; niter++)
        {
            Apply(expH,mps_compressor);
            ReCenter(2);
            Apply(expH,mps_compressor);
            ReCenter(1);
            double Enew=GetExpectation(H)/(itsL-1);
            double dE=Enew-E1;
            E1=Enew;
            Logger->LogInfoV(3,"E=%.9f, dE=%.2e, niter=%4d",E1,dE,niter);
            if (fabs(dE)<=isl.itsEps.itsDelatEnergy1Epsilon || dE>0.0) break;
        }
        Orthogonalize(mps_compressor);
        E1=GetExpectation(H)/(itsL-1);
        Logger->LogInfoV(1,"End %4d iterations, dt=%.3f,   E=%.9f, D=%4d, isl.Dmax=%4d",niter,isl.itsdt,E1,D,isl.itsDmax);
    }
    delete mps_compressor;
    return E1;
}

//-----------------------------------------------------------------------------
//
//  THis follows PHYSICAL REVIEW B 78, 155117 2008 figure 14
//
void iTEBDStateImp::Apply(const Matrix4RT& expH,SVCompressorC* comp)
{
    assert(comp);
//    assert(TestOrthogonal(1e-9));
    //
    //  Make sure everything is square
    assert(s1.siteA->GetD2()==s1.siteB->GetD1());
    assert(s1.siteA->GetD1()==s1.siteB->GetD2());
    assert(s1.siteA->GetD1()==s1.siteA->GetD2());
    int D=s1.siteA->GetD1();

    dVectorT  Thetap(itsd*itsd);
    for (int n=0;n<itsd*itsd;n++)
    {
        Thetap[n].SetLimits(D,D);
        Fill(Thetap[n],dcmplx(0.0));
    }

    for (int mb=0; mb<itsd; mb++)
    for (int ma=0; ma<itsd; ma++)
    {
        MatrixCT theta13  =GammaA()[ma]*lambdaA()*GammaB()[mb];  //Figure 14 v
        int nab=0;
        for (int nb=0; nb<itsd; nb++)
        for (int na=0; na<itsd; na++,nab++)
            Thetap[nab]+= theta13*expH(ma,na,mb,nb);
    }

    DiagonalMatrixRT lb=lambdaB();
    auto [gammap,lambdap]=Orthogonalize(Thetap,lb);
    UnpackOrthonormal(gammap,lambdap,comp);
}


}
