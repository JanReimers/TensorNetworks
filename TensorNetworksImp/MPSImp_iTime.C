#include "TensorNetworksImp/MPSImp.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/TNSLogger.H"
//#include <iostream>
//#include <iomanip>

//using std::cout;
//using std::endl;

namespace TensorNetworks
{

double MPSImp::FindiTimeGroundState(const Hamiltonian* H,const IterationSchedule& is)
{
    double E1=0;
    itsLogger->LogInfoV(1,"Initiate iTime GS iterations, Dmax=%4d",GetMaxD());
    for (is.begin(); !is.end(); is++)
        E1=FindiTimeGroundState(H,*is);

    MPO* H2=H->CreateH2Operator();
    double E2=GetExpectation(H2);
    delete H2;
    itsLogger->LogInfoV(0,"Finished iTime GS iterations D=%4d, <E>=%.9f, <E^2>-<E>^2=%.2e",GetMaxD(),E1,E2);
    return E2-E1*E1;
}

double MPSImp::FindiTimeGroundState(const Hamiltonian* H,const IterationScheduleLine& isl)
{
    assert(isl.itsDmax>0 || isl.itsEps.itsMPSCompressEpsilon>0);

    double E1=GetExpectation(H)/(itsL-1);

    SVCompressorC* mps_compressor =Factory::GetFactory()->MakeMPSCompressor(isl.itsDmax,isl.itsEps.itsMPSCompressEpsilon);
    SVCompressorR* mpo_compressor =Factory::GetFactory()->MakeMPOCompressor(0          ,isl.itsEps.itsMPOCompressEpsilon);
    MPO* W =H->CreateOperator(isl.itsdt,isl.itsTrotterOrder);
    W->Compress(mpo_compressor);
//    W->Report(cout);
    itsLogger->LogInfoV(1,"   Begin iterations, dt=%.3f,  Dw=%4d, GetMaxD=%4d, isl.Dmax=%4d, epsMPO=%.1e, epsMPS=%.1e ",isl.itsdt,W->GetMaxDw(),GetMaxD(),isl.itsDmax,isl.itsEps.itsMPOCompressEpsilon,isl.itsEps.itsMPSCompressEpsilon);
    int niter=1;
    for (; niter<isl.itsMaxGSSweepIterations; niter++)
    {

        ApplyInPlace(W); //Make copy of the |Psi2>=H|Psi> which now has large D_2 = D_1*Dw
        MPS* Psi2=Clone();
        //
        //  Compress in both directions
        //
        NormalizeAndCompress(DLeft ,mps_compressor); //no Dmax enforced
        NormalizeAndCompress(DRight,mps_compressor); //Enforce Dmax on this one. It will spit out warning if we hit the Dmax limit.
        Psi2->Normalize(DLeft);
        Psi2->Normalize(DRight); //Sweeping both directions seems to give a massive speedup.
        //
        // Now optimize this to be as close as possible to Psi2
        //
        int niter=Optimize(Psi2,isl);
        //
        //  Check energy convergence
        //
        double Enew=GetExpectation(H)/(itsL-1);
        double dE=Enew-E1;
        E1=Enew;
        if (fabs(dE)<=isl.itsEps.itsDelatEnergy1Epsilon) break;
        itsLogger->LogInfoV(2,"      E=%.9f, dE=%.2e, D1=%4d, D2=%4d, <2|1> niter=%4d",E1,dE,GetMaxD(),Psi2->GetMaxD(),niter);
        delete Psi2;
    }
    itsLogger->LogInfoV(1,"     End iterations, dt=%.3f, niter=%4d, D=%4d, E=%.9f.",isl.itsdt,niter,GetMaxD(),E1);
    delete mpo_compressor;
    delete mps_compressor;
    return E1;
}

//--------------------------------------------------------------------------------------
//
//  Vary this MPS to be as close as possible to Psi2 by minimizing ||this-Psi2||^2
//
int MPSImp::Optimize(const MPS* Psi2,const IterationScheduleLine& isl)
{
    LoadCaches(Psi2);

    int in=0;
    for (;in<isl.itsMaxOptimizeIterations; in++)
    {
        double O1=Sweep(DLeft ,Psi2);  //This actually sweeps to the right, but leaves left normalized sites in its wake
        double O2=Sweep(DRight,Psi2); //return ||psi2-psi1||^2
        double dO=fabs(O2-O1);
        itsLogger->LogInfoV(3,"         Minimize ||psi2-psi1||^2 err1=%.1e, err2=%.1e, delta=%.1e",O1,O2,dO);
        if (dO<=isl.itsEps.itsDelatNormEpsilon) break;
    }
    return in+1;
}

double MPSImp::Sweep(Direction lr,const MPS* Psi2)
{
    int iter=0;
    const MPSImp* psi2Imp=dynamic_cast<const MPSImp*>(Psi2);
    assert(psi2Imp);
//    MatrixCT MTrace(1,1);
//    MTrace(1,1)=dcmplx(1.0);
    ForLoop(lr)
    {
        CheckSiteNumber(ia);
        assert(IsRLNormalized(ia));
        assert(GetRLCache(DLeft ,ia-1)==Calc12Left_Cache(Psi2,ia,false));
        assert(GetRLCache(DRight,ia+1)==Calc12RightCache(Psi2,ia,false));

        itsSites[ia]->Optimize(psi2Imp->itsSites[ia],
                               GetRLCache(DLeft,ia-1),
                               GetRLCache(DRight,ia+1));
//        MTrace=itsSites[ia]->IterateF(lr,MTrace);
        NormalizeSite(lr,ia);
        itsSites[ia]->UpdateCache(psi2Imp->itsSites[ia],
                                  GetRLCache(DLeft,ia-1),
                                  GetRLCache(DRight,ia+1));

        iter++; //ia doesn;t always count upwards, but this guy does.
    }
    double O22=1.0; //Psi2->GetOverlap(Psi2);
    double O21=Psi2->GetOverlap(this);
    double O12=O21;
    double O11=1.0;
    double dO=O11-O12-O21+O22;

//    assert(MTrace.GetNumRows()==1);
//    assert(MTrace.GetNumCols()==1);
//    double IM=imag(MTrace(1,1));
//    if (fabs(IM)>1e-10)
//        cout << "Warning: MatrixProductState::Sweep Imag(M)=" << IM << endl;
//    return 1.0-real(MTrace(1,1)); //Not working
    return dO;
}

MatrixCT MPSImp::GetRLCache (Direction lr,int isite) const
{
    //CheckSiteNumber(isite); this function accepts out of range site numbers
    MatrixCT RL(1,1);
    RL(1,1)=dcmplx(1.0);
    if (isite>=1 && isite<=itsL)  RL=itsSites[isite]->GetRLCache(lr);
    return RL;
}

void MPSImp::LoadCaches(const MPS* Psi2)
{
    Calc12Left_Cache(Psi2,itsL,true);
    Calc12RightCache(Psi2,   1,true);
}

//
//  Calculate  L&R caches for <psi2|psi1> contractions.
//
MatrixCT MPSImp::Calc12Left_Cache(const MPS* psi2,int isite, bool cache) const
{
//    CheckSiteNumber(isite);  this function accepts out of bounds site numbers
    const MPSImp* psi2Imp=dynamic_cast<const MPSImp*>(psi2);
    assert(psi2Imp);

    MatrixCT F(1,1);
    F(1,1)=dcmplx(1.0);
    for (int ia=1; ia<isite; ia++)
        F=itsSites[ia]->IterateLeft_F(psi2Imp->itsSites[ia],F,cache);
    return F;
}
MatrixCT MPSImp::Calc12RightCache(const MPS* psi2,int isite, bool cache) const
{
//    CheckSiteNumber(isite); this function accepts out of bounds site numbers
    const MPSImp* psi2Imp=dynamic_cast<const MPSImp*>(psi2);
    assert(psi2Imp);
    MatrixCT F(1,1);
    F(1,1)=dcmplx(1.0);
    for (int ia=itsL; ia>isite; ia--)
        F=itsSites[ia]->IterateRightF(psi2Imp->itsSites[ia],F,cache);
    return F;
}





void  MPSImp::ApplyInPlace(const Operator* o)
{
    SiteLoop(ia)
        itsSites[ia]->ApplyInPlace(o->GetSiteOperator(ia));
}

MPS*  MPSImp::Apply(const Operator* o) const
{
    MPSImp* psiPrime=new MPSImp(itsL,itsS,GetMaxD(),itsNormEps,itsLogger);
    SiteLoop(ia)
        itsSites[ia]->Apply(o->GetSiteOperator(ia),psiPrime->itsSites[ia]);

    return psiPrime;
}

}; // namespace
