#include "TensorNetworksImp/iTEBDStateImp.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/TNSLogger.H"

#include "TensorNetworksImp/Bond.H"

#include "NumericalMethods/ArpackEigenSolver.H"
#include "NumericalMethods/PrimeEigenSolver.H"
#include "NumericalMethods/LapackEigenSolver.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include "oml/cnumeric.h"
#include <iomanip>

namespace TensorNetworks
{


double iTEBDStateImp::FindiTimeGroundState(const Hamiltonian* H,const IterationSchedule& is)
{
    assert(Logger); //Make sure we have global logger.
    Canonicalize(TensorNetworks::DLeft);
    Logger->LogInfoV(0,"Initiate iTime GS iterations, D=%4d, Norm status=%s",GetMaxD(),GetNormStatus().c_str());
    Logger->LogInfo(1,"     dt    epsE      D  Dmax   Dw niter   E        Ortho eror");
    MPO* H2=H->CreateH2Operator();

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
//    MPO* expH=H->CreateOperator(dt,TensorNetworks::SecondOrder);
//    MPO* expH=H->CreateiMPO(dt,TensorNetworks::FirstOrder);

    int Dmax=GetMaxD();
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
            //
            //  Warm the state, i.e. get some real numbers in the matrix areas
            //
            Apply(expH,mps_compressor);
            ReCenter(2);
            Apply(expH,mps_compressor);
            ReCenter(1);
        }
        Logger->LogInfo(2,"      Dw       E          dE      niter    Ortho errors");
        double oerr1,oerr2;
        int niter=1;
        for (; niter<=isl.itsMaxGSSweepIterations; niter++)
        {
            Apply(expH,mps_compressor);
//            oerr1=ApplyOrtho(expH,mps_compressor,dt*dt/100,nOrthIter);
            ReCenter(2);
            Apply(expH,mps_compressor);
//            oerr2=ApplyOrtho(expH,mps_compressor,dt*dt/100,nOrthIter);
            ReCenter(1);
            double Enew=GetExpectation(H)/(itsL-1);
            double dE=Enew-E1;
            E1=Enew;
            Logger->LogInfoV(2,"%4d %.9f %.2e %4d     %.1e/%.1e",H->GetMaxDw(),E1,dE,niter,oerr1, oerr2);
            if (fabs(dE)<=isl.itsEps.itsDelatEnergy1Epsilon /*|| dE>0.0*/) break;
        }
        oerr1=OrthogonalizeI(mps_compressor,1e-13,nOrthIter);
        E1=GetExpectation(H);
        double E2=GetExpectation(H2);
        E2= E2-E1*E1;
       Logger->LogInfoV(1,"%.3f %.2e %4d %4d %4d %4d %.9f %.1e",
                        isl.itsdt,isl.itsEps.itsDelatEnergy1Epsilon,D,isl.itsDmax,H->GetMaxDw(),niter,E1,oerr1);
    }
    delete mps_compressor;
//    delete expH;
    return E1;
}

DiagonalMatrixRT Extend(const DiagonalMatrixRT& lambda,const MPO* o)
{
    // Extract Dw from the MPO
    assert(o->GetL()==2);
    const SiteOperator* soA=o->GetSiteOperator(1);
    const SiteOperator* soB=o->GetSiteOperator(2);
    assert(soA->GetDw12().Dw2==soB->GetDw12().Dw1);
    assert(soA->GetDw12().Dw1==soB->GetDw12().Dw2);
    int Dw=soA->GetDw12().Dw1; //Same as Dw3
    //
    //  Now load Dw copies of lb into the extended lambdaB
    //
    int D=lambda.GetDiagonal().size();
    DiagonalMatrixRT extended_lambda(D*Dw);
    int iw=1;
    for (int w=1;w<=Dw;w++)
        for (int i=1;i<=D;i++,iw++)
            extended_lambda(iw)=lambda(i,i);

    return extended_lambda;
}


double iTEBDStateImp::ApplyOrtho(const Matrix4RT& expH,SVCompressorC* comp,double eps,int maxIter)
{
    assert(comp);
    dVectorT gamma=ContractTheta(expH);
    DiagonalMatrixRT lambda=lambdaB();
    OrthogonalizeI(gamma,lambda,eps,maxIter);
    s1.bondB->SetSingularValues(lambda,0.0);
    return UnpackOrthonormal(gamma,comp);
}

double iTEBDStateImp::ApplyOrtho(const MPO* expH,SVCompressorC* comp,double eps,int maxIter)
{
    assert(comp);
    dVectorT          gamma=ContractTheta(expH);
    DiagonalMatrixRT lambda=Extend(lambdaB(),expH);
    OrthogonalizeI(gamma,lambda,eps,maxIter);
    double truncationError=comp->Compress(gamma,lambda); //Reduce form DDw to D.
    s1.bondB->SetSingularValues(lambda,truncationError); //Don't use lambdap any more in case it is not normalized
    return UnpackOrthonormal(gamma,comp);
}

//-----------------------------------------------------------------------------
//
//  THis follows PHYSICAL REVIEW B 78, 155117 2008 figure 14
//
double iTEBDStateImp::Apply(const Matrix4RT& expH,SVCompressorC* comp,bool orthogonalize)
{
    assert(comp);
    dVectorT gamma=ContractTheta(expH);
    if (orthogonalize)
    {
        DiagonalMatrixRT lambda=lambdaB();
        Orthogonalize(gamma,lambda);
        s1.bondB->SetSingularValues(lambda,0.0);
    }
    return UnpackOrthonormal(gamma,comp);
}


double iTEBDStateImp::Apply(const MPO* expH,SVCompressorC* comp,bool orthogonalize)
{
    assert(comp);
    DiagonalMatrixRT lambda=Extend(lambdaB(),expH);
    dVectorT gamma=ContractTheta(expH);
    if (orthogonalize) OrthogonalizeI(gamma,lambda,1e-13,100);
    double truncationError=comp->Compress(gamma,lambda); //Reduce form DDw to D.
    s1.bondB->SetSingularValues(lambda,truncationError); //Don't use lambdap any more in case it is not normalized
    return UnpackOrthonormal(gamma,comp);;
}


}
