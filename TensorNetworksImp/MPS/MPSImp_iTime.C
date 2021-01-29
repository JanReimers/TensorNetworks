#include "TensorNetworksImp/MPS/MPSImp.H"
#include "TensorNetworksImp/MPS/Bond.H"
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
    assert(Logger); //Make sure we have global logger.
    double E1=0;
    Logger->LogInfoV(1,"Initiate iTime GS iterations, Dmax=%4d",GetMaxD());
    for (is.begin(); !is.end(); is++)
        E1=FindiTimeGroundState(H,*is);

    MPO* H2=H->CreateH2Operator();
    double E2=GetExpectation(H2);
    E2=E2-E1*E1;
    E1/=itsL-1;
    delete H2;
    Logger->LogInfoV(0,"Finished iTime GS iterations D=%4d, <E>=%.9f, <E^2>-<E>^2=%.2e",GetMaxD(),E1,E2);
    return E2;
}

double MPSImp::FindiTimeGroundState(const Hamiltonian* H,const IterationScheduleLine& isl)
{
    assert(isl.itsDmax>0 || isl.itsEps.itsMPSCompressEpsilon>0);

    double E1=GetExpectation(H);

    SVCompressorC* mps_compressor =Factory::GetFactory()->MakeMPSCompressor(isl.itsDmax,isl.itsEps.itsMPSCompressEpsilon);
    SVCompressorR* mpo_compressor =Factory::GetFactory()->MakeMPOCompressor(0          ,isl.itsEps.itsMPOCompressEpsilon);
    MPO* W =H->CreateOperator(isl.itsdt,isl.itsTrotterOrder,TensorNetworks::Std,isl.itsEps.itsMPOCompressEpsilon);
    W->Compress(TensorNetworks::Std,mpo_compressor);
//    W->Report(cout);
//    Logger->LogInfoV(1,"   Begin iterations, dt=%.3f,  Dw=%4d, GetMaxD=%4d, isl.Dmax=%4d, epsMPO=%.1e, epsMPS=%.1e ",isl.itsdt,W->GetMaxDw(),GetMaxD(),isl.itsDmax,isl.itsEps.itsMPOCompressEpsilon,isl.itsEps.itsMPSCompressEpsilon);
    Logger->LogInfo(2,"iter oiter      E          dE    dlambda    D1   D2");
    int niter=1;
    int maxoiter=0;
    for (; niter<isl.itsMaxGSSweepIterations; niter++)
    {
        BondsType bond_cache(itsBonds); //Clone all the bonds
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
        int noiter=Optimize(Psi2,isl);
        maxoiter=Max(maxoiter,noiter);
        //
        //  Check energy convergence
        //
        double Enew=GetExpectation(H);
        double dE=Enew-E1;
        E1=Enew;
        double dl=GetMaxDeltal(bond_cache); //Max delta lambda on bonds.
        Logger->LogInfoV(2,"%4d %4d %.11f %8.1e %.1e %4d %4d",niter,noiter,E1/(itsL-1),dE,dl,dE,GetMaxD(),Psi2->GetMaxD());
        if (fabs(dE)<=isl.itsEps.itsDelatEnergy1Epsilon &&  dl<isl.itsEps.itsDeltaLambdaEpsilon) break;
        if (dE>0.0 && niter>1) break;
        delete Psi2;
    }
    Logger->LogInfoV(1,"     End iterations, dt=%.3f, niter=%4d, oiter=%4d, D=%4d, E=%.9f.",isl.itsdt,niter,maxoiter,GetMaxD(),E1/(itsL-1));
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
    for (;in<isl.itsMaxOptimizeIterations;)
    {
        double O1=Sweep(DLeft ,Psi2);  //This actually sweeps to the right, but leaves left normalized sites in its wake
        double O2=Sweep(DRight,Psi2); //return ||psi2-psi1||^2
        double dO=fabs(O2-O1);
        in++;
        Logger->LogInfoV(3,"         Minimize ||psi2-psi1||^2 err1=%.1e, err2=%.1e, delta=%.1e",O1,O2,dO);
        if (dO<=isl.itsEps.itsDelatNormEpsilon) break;
    }
    return in;
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




//
//  |this> = O*|this>
//
void  MPSImp::ApplyInPlace(const MPO* o)
{
    SiteLoop(ia)
        itsSites[ia]->ApplyInPlace(o->GetSiteOperator(ia));
}


}; // namespace
