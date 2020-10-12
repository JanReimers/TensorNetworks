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
    assert(this->itsDmax>0);
    for (is.begin(); !is.end(); is++)
        E1=FindiTimeGroundState(H,*is);

    MPO* H2=H->CreateH2Operator();
    double E2=GetExpectation(H2);
    delete H2;
    return E2-E1*E1;
}

double MPSImp::FindiTimeGroundState(const Hamiltonian* H,const IterationScheduleLine& isl)
{
    assert(isl.itsDmax>0 || isl.itsEps.itsMPSCompressEpsilon>0);
    SVCompressorC* mps_compressor=Factory::GetFactory()->MakeMPSCompressor(isl.itsDmax,isl.itsEps.itsMPSCompressEpsilon);
    SVCompressorR* mpo_compressor=Factory::GetFactory()->MakeMPOCompressor(0          ,isl.itsEps.itsMPOCompressEpsilon);

    double E1=GetExpectation(H)/(itsL-1);
//    cout << isl << endl;
//    cout.precision(5);
//    cout << "E=" << std::fixed << E1 << endl;
    MPO* W =H->CreateOperator(isl.itsdt,isl.itsTrotterOrder);
    W->Compress(mpo_compressor);
//    double percent=W->Compress(0,isl.itsEps.itsMPOCompressEpsilon);
//    cout << "FindGroundState dt=" << isl.itsdt << " " << percent << "% compresstion" << endl;
    int niter=1;
    assert(this->itsDmax>0);
    for (; niter<isl.itsMaxGSSweepIterations; niter++)
    {
        ApplyInPlace(W); //this now has large D_2 = D_1*Dw
        MPS* Psi2=Clone(); //Make copy of the uncompressed Psi
        //
        //  Compress in both directions
        //
        NormalizeAndCompress(DLeft ,mps_compressor);
        assert(this->itsDmax>0);
        NormalizeAndCompress(DRight,mps_compressor);
        //
        // Now optimise this to be as close as possible to Psi2
        //
        Optimize(Psi2,isl);
        //
        //  Check energy convergence
        //
        double Enew=GetExpectation(H)/(itsL-1);
        delete Psi2;
        double dE=Enew-E1;
        //cout << "n=" << niter << "  E=" << std::fixed << Enew << std::scientific << "  dE=" << dE << endl;
        E1=Enew;
        if (fabs(dE)<=isl.itsEps.itsDelatEnergy1Epsilon) break;
    }
    cout << "Niter,dt,E = " << niter << " " << isl.itsdt << " " << std::fixed << E1 << endl;
    delete mpo_compressor;
    delete mps_compressor;
    return E1;
}

//--------------------------------------------------------------------------------------
//
//  Vary this MPS to be as close as possible to Psi2 by minimizing ||this-Psi2||^2
//
void MPSImp::Optimize
(const MPS* Psi2,const IterationScheduleLine& isl)
{
    LoadCaches(Psi2);

    for (int in=0; in<isl.itsMaxOptimizeIterations; in++)
    {
        itsLogger->LogInfo(0,"Sweep Right");
        double O1=Sweep(DLeft,Psi2);  //This actually sweeps to the right, but leaves left normalized sites in its wake

//        cout << "Left  " << in << " Norm error=" << O1 << endl;
        itsLogger->LogInfo(0,"Sweep Left");
        double O2=Sweep(DRight,Psi2);
//        cout << "Right " << in << " Norm error=" << O2 << endl;
        //cout << "Norm change=" << O2-O1 << endl;
        if (fabs(O2-O1)<=isl.itsEps.itsDelatNormEpsilon) break;
    }
    //        double O22=Psi2->GetOverlap(Psi2);
//        double O21=Psi2->GetOverlap(Psi1);
//        double O12=Psi1->GetOverlap(Psi2);
//        double O11=Psi1->GetOverlap(Psi1);
//        cout << "O11 O12 O21 O22 delta=" << O11 << " " << O12 << " " << O21 << " " << O22 << " " << O11-O12-O21+O22 << endl;
//

}

double MPSImp::Sweep(Direction lr,const MPS* Psi2)
{
    int iter=0;
    const MPSImp* psi2Imp=dynamic_cast<const MPSImp*>(Psi2);
    assert(psi2Imp);
    MatrixCT MTrace(1,1);
    MTrace(1,1)=dcmplx(1.0);
    ForLoop(lr)
    {
        CheckSiteNumber(ia);
//        cout << "----- Opimizing site " << ia << " " << GetNormStatus() << " -----" << endl;

        assert(IsRLNormalized(ia));
        assert(GetRLCache(DLeft ,ia-1)==CalcHeffLeft(Psi2,ia,false));
        assert(GetRLCache(DRight,ia+1)==CalcHeffRight(Psi2,ia,false));
        itsSites[ia]->Optimize(psi2Imp->itsSites[ia],
                               GetRLCache(DLeft,ia-1),
                               GetRLCache(DRight,ia+1));
        MTrace=itsSites[ia]->IterateF(lr,MTrace);
        NormalizeSite(lr,ia);
        itsSites[ia]->UpdateCache(psi2Imp->itsSites[ia],
                                  GetRLCache(DLeft,ia-1),
                                  GetRLCache(DRight,ia+1));

        iter++; //ia doesn;t always count upwards, but this guy does.
    }
//    double O11=GetOverlap(this);
//    double O12=GetOverlap(Psi2);
//    cout << "<psi1|psi1> <psi1|psi2>, delta=" << O11 << " " << O12 << " " << O12-O11 << endl;
//    cout << "MTrace=" << MTrace << endl;
    assert(MTrace.GetNumRows()==1);
    assert(MTrace.GetNumCols()==1);
    double IM=imag(MTrace(1,1));
    if (fabs(IM)>1e-10)
        cout << "Warning: MatrixProductState::Sweep Imag(M)=" << IM << endl;
    return 1.0-real(MTrace(1,1));
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
    CalcHeffLeft(Psi2,itsL,true);
    CalcHeffRight(Psi2,   1,true);
}





void  MPSImp::ApplyInPlace(const Operator* o)
{
    SiteLoop(ia)
    itsSites[ia]->ApplyInPlace(o->GetSiteOperator(ia));
}

MPS*  MPSImp::Apply(const Operator* o) const
{
    MPSImp* psiPrime=new MPSImp(itsL,itsS,1,itsNormEps,itsLogger);
    SiteLoop(ia)
    {
//        cout << "------------- Site " << ia << "-----------------" << endl;
        itsSites[ia]->Apply(o->GetSiteOperator(ia),psiPrime->itsSites[ia]);
    }

    return psiPrime;
}

}; // namespace
