#include "TensorNetworksImp/iTEBD/iTEBDStateImp.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/TNSLogger.H"

#include "TensorNetworksImp/MPS/Bond.H"

#include "NumericalMethods/ArpackEigenSolver.H"
#include "NumericalMethods/PrimeEigenSolver.H"
#include "NumericalMethods/LapackEigenSolver.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include "Containers/ptr_vector.h"
#include "oml/cnumeric.h"
#include <iomanip>

namespace TensorNetworks
{


double iTEBDStateImp::FindiTimeGroundState(const Hamiltonian* H,const IterationSchedule& is)
{
    assert(Logger); //Make sure we have global logger.
    Canonicalize(TensorNetworks::DLeft);
    Logger->LogInfoV(0,"Initiate iTime GS iterations, D=%4d, Norm status=%s",GetMaxD(),GetNormStatus().c_str());
    Logger->LogInfo(1,"     dt    epsE      D  Dmax   Dw niter   E        Ortho error dE  DlambdaA DlambdaB");
    iMPO* H2=H->CreateiH2Operator();

    double E1=0;
    for (is.begin(); !is.end(); is++)
        E1=FindiTimeGroundState(H,H2,*is);

    double E2=GetExpectation(H2)/(itsL-1)/(itsL-1);
    E2= E2-E1*E1;
    delete H2;
    Logger->LogInfoV(0,"Finished iTime GS iterations D=%4d, <E>=%.9f, <E^2>-<E>^2=%.2e",GetMaxD(),E1,E2);
    return E2;
}

double iTEBDStateImp::FindiTimeGroundState(const Hamiltonian* H,const iMPO* H2,const IterationScheduleLine& isl)
{
    assert(isl.itsDmax>0 || isl.itsEps.itsMPSCompressEpsilon>0);
    double dt=isl.itsdt;

    InitGates(H,dt,isl.itsTrotterOrder,isl.itsEps.itsMPOCompressEpsilon);

    int Dmax=GetMaxD();
    double epsCompress=1e-16;
    SVCompressorC* mps_compressor =Factory::GetFactory()->MakeMPSCompressor(Dmax,epsCompress);
    int nOrthIter=100;
    double epsOrth= dt==0.0 ? 1e-13 : dt*dt/1000.0;
    assert(epsOrth>0.0);
    OrthogonalizeI(mps_compressor,epsOrth,nOrthIter);
    ReCenter(1);
    double E1=GetExpectationDw1(H)/(itsL-1);
    double dE=0.0;

    for (int D=Dmax;D<=isl.itsDmax;D+=isl.itsDeltaD)
    {
        if (D>Dmax)
        {
            IncreaseBondDimensions(D);
            delete mps_compressor;
            mps_compressor =Factory::GetFactory()->MakeMPSCompressor(D,epsCompress);
        }
        Logger->LogInfo(2,"      Dw       E          dE      niter    Ortho errors  DlambdaA DlambdaB");
        double oerr1,oerr2,dA,dB;
        int niter=1;
        for (; niter<=isl.itsMaxGSSweepIterations; niter++)
        {
            DiagonalMatrixRT lA=lambdaA();
            DiagonalMatrixRT lB=lambdaB();

            Apply(mps_compressor,1);
            ReCenter(1);
            dA=Max(fabs(lA-lambdaA()));
            dB=Max(fabs(lB-lambdaB()));

            Logger->LogInfoV(2,"%4d %.9f %.2e %4d     %.1e/%.1e  %.1e  %.1e",H->GetMaxDw(),E1,dE,niter,oerr1, oerr2, dA, dB);
            if (
                 dA <=isl.itsEps.itsDeltaLambdaEpsilon &&
                 dB <=isl.itsEps.itsDeltaLambdaEpsilon
                ) break;
        }
        ReCenter(1);
        oerr1=OrthogonalizeI(mps_compressor,epsOrth,nOrthIter);
//        ReCenter(2);
//        oerr2=OrthogonalizeI(mps_compressor,epsOrth,nOrthIter);
        double Enew=GetExpectationDw1(H);
        dE=Enew-E1;
        E1=Enew;
        double E2=1.0;//GetExpectation(H2);
        E2= E2-E1*E1;
       Logger->LogInfoV(1,"%.3f %.2e %4d %4d %4d %4d %.9f %.1e  %.1e  %.1e  %.1e",
                        isl.itsdt,isl.itsEps.itsDelatEnergy1Epsilon,D,
                        isl.itsDmax,H->GetMaxDw(),niter,E1,oerr1,dE,dA,dB);
    }
    delete mps_compressor;
    return E1;
}

//
//  Apply single impo gate and compress at the end
//
void iTEBDStateImp::Apply(const Multi_iMPOType& expH,SVCompressorC* D_compressor,int center)
{
    ReCenter(center);
    dVectorT theta_BA=ContractTheta(expH[0],lBlAl); //is root(lA)*GB*lB*GA*root(lA) better?
    Unpack_iMPO(theta_BA,D_compressor);
}

//
//  Apply multiple gates and compress at the end
//
void iTEBDStateImp::Apply(const MultigateType& expH,SVCompressorC* comp,int center)
{
    SVCompressorC* eps_compressor =Factory::GetFactory()->MakeMPSCompressor(0,1e-7);
    int Ngate=expH.size();
    assert(Ngate>0);
    int isite=center;
    int igate=1; //We need to know when the second last gate occurs.
    for (auto& gate:expH)
    {
        ReCenter(isite++);
        dVectorT gamma=ContractTheta(gate,lAlBl);
        if (igate++<=Ngate-2)
            Unpack(gamma,eps_compressor); //only eps compression before the last two gates.
        else
            Unpack(gamma,comp); //Compression after the last two gates. Each compression truncates a different bond.
    }
}

//
//  Apply multiple mpo gates and compress at the end
//
//void iTEBDStateImp::Apply(const MultiMPOType& expH,SVCompressorC* comp,int center)
//{
//    SVCompressorC* eps_compressor =Factory::GetFactory()->MakeMPSCompressor(0,1e-13);
//    int Ngate=expH.size();
//    assert(Ngate>0);
//    int isite=center;
//    int igate=1; //We need to know when the second last gate occurs.
//    for (auto& gate:expH)
//    {
//        ReCenter(isite++);
//        dVectorT gamma=ContractTheta(&gate,lAlBl);
//        if (igate++<=Ngate-2)
//            Unpack(gamma,eps_compressor); //only eps compression before the last two gates.
//        else
//            Unpack(gamma,comp); //Compression after the last two gates. Each compression truncates a different bond.
//    }
//}
//

DiagonalMatrixRT Extend(const DiagonalMatrixRT& lambda,const iMPO* o)
{
    // Extract Dw from the MPO
    assert(o->GetL()==2);
    const SiteOperator* soA=o->GetSiteOperator(1);
#if DEBUG
    const SiteOperator* soB=o->GetSiteOperator(2);
    assert(soA->GetDw12().Dw2==soB->GetDw12().Dw1);
    assert(soA->GetDw12().Dw1==soB->GetDw12().Dw2);
#endif
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
    dVectorT gamma=ContractTheta(expH,AlB);
    DiagonalMatrixRT lambda=lambdaB();
    OrthogonalizeI(gamma,lambda,eps,maxIter);
    s1.bondB->SetSingularValues(lambda,0.0);
    return UnpackOrthonormal(gamma,comp);
}

double iTEBDStateImp::ApplyOrtho(const iMPO* expH,SVCompressorC* comp,double eps,int maxIter)
{
    assert(comp);
    dVectorT          gamma=ContractTheta(expH,AlB);
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
    dVectorT gamma=ContractTheta(expH,AlB);
    if (orthogonalize)
    {
        DiagonalMatrixRT lambda=lambdaB();
        Orthogonalize(gamma,lambda);
        s1.bondB->SetSingularValues(lambda,0.0);
    }
    return UnpackOrthonormal(gamma,comp);
}


double iTEBDStateImp::Apply(const iMPO* expH,SVCompressorC* comp,bool orthogonalize)
{
    assert(comp);
    DiagonalMatrixRT lambda=Extend(lambdaB(),expH);
    dVectorT gamma=ContractTheta(expH,AlB);
    if (orthogonalize) OrthogonalizeI(gamma,lambda,1e-13,100);
    double truncationError=comp->Compress(gamma,lambda); //Reduce form DDw to D.
    s1.bondB->SetSingularValues(lambda,truncationError); //Don't use lambdap any more in case it is not normalized
    return UnpackOrthonormal(gamma,comp);;
}


}
