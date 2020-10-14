#include "TensorNetworksImp/MPSImp.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/TNSLogger.H"
#include "Containers/Matrix6.H"

//--------------------------------------------------------------------------------------
//
// Find variational ground state
//
namespace TensorNetworks
{

double MPSImp::FindVariationalGroundState(const Hamiltonian* H,const IterationSchedule& is)
{
    itsLogger->ReadyToStart("Right normalize");
    Normalize(DRight);
    itsLogger->LogInfo(3,"Load L&R caches");
    LoadHeffCaches(H);

    double DE2=0;
    for (is.begin(); !is.end(); is++)
        DE2=FindVariationalGroundState(H,*is);
    return DE2;
}

double MPSImp::FindVariationalGroundState(const Hamiltonian* H, const IterationScheduleLine& isl)
{
    int Dmax=GetMaxD();
    MPO* H2=H->CreateH2Operator();
    double DE2=0;
    for (int D=Dmax;D<=isl.itsDmax;D+=isl.itsDeltaD)
    {
        if (D>Dmax)
        {
            IncreaseBondDimensions(D);
            Normalize(DRight);
            LoadHeffCaches(H);
        }
        int in=0;
        for (; in<isl.itsMaxGSSweepIterations; in++)
        {
            itsLogger->LogInfo(1,"Sweep Right");
            Sweep(DLeft,H,isl.itsEps);  //This actually sweeps to the right, but leaves left normalized sites in its wake
            itsLogger->LogInfo(1,"Sweep Left");
            Sweep(DRight,H,isl.itsEps);
            double dE=GetMaxDeltaE();
            //cout << "dE=" << dE << endl;
            if (dE<isl.itsEps.itsDelatEnergy1Epsilon) break;
        }
        in++;

        double E1=GetExpectation(H);
        double E2=GetExpectation(H2);
        DE2=E2-E1*E1;
        itsLogger->LogInfoV(0,"Variational GS D=%4d, %4d iterations, <E>=%.9f, <E^2>-<E>^2=%.2e",D,in,E1,DE2);
        IterationEnergy(E1);
    }
    delete H2;
    return DE2;
}

void MPSImp::Sweep(Direction lr,const Hamiltonian* h,const Epsilons& eps)
{
    int iter=0;
    ForLoop(lr)
    {
        CheckSiteNumber(ia);
        Refine(lr,h,eps,ia);
        double de=fabs(itsSites[ia]->GetIterDE());
        IterationDeltaE(iter,de);
        iter++; //ia doesn;t always count upwards, but this guy does.
    }
    itsNSweep++;
}


void MPSImp::Refine(Direction lr,const Hamiltonian *h,const Epsilons& eps,int isite)
{
//    assert(CheckNormalized(isite,eps.itsNormalizationEpsilon));
    CheckSiteNumber(isite);
    assert(IsRLNormalized(isite));
    if (!itsSites[isite]->IsFrozen())
    {
        itsLogger->LogInfo(2,isite,"Calculating Heff"); //Logger will update the graphs
        Matrix6CT Heff6=GetHeffIterate(h,isite); //New iterative version
        itsLogger->LogInfo(2,isite,"Running eigen solver"); //Logger will update the graphs
        itsSites[isite]->Refine(Heff6.Flatten(),eps);
    }
    UpdateEnergyData(isite);
    NormalizeSite(lr,isite);
    itsSites[isite]->UpdateCache(h->GetSiteOperator(isite),
                                 GetHeffCache(DLeft,isite-1),
                                 GetHeffCache(DRight,isite+1));

}

Vector3CT MPSImp::GetHeffCache (Direction lr,int isite) const
{
    //CheckSiteNumber(isite); this function accepts out of range site numbers
    Vector3CT H(1,1,1,1);
    H(1,1,1)=dcmplx(1.0);
    if (isite>=1 && isite<=itsL)  H=itsSites[isite]->GetHeffCache(lr);
    return H;
}


Matrix6CT MPSImp::GetHeffIterate   (const Hamiltonian* h,int isite) const
{
    CheckSiteNumber(isite);
    Vector3CT Lcache=GetHeffCache(DLeft,isite-1);
    Vector3CT Rcache=GetHeffCache(DRight,isite+1);
    return itsSites[isite]->GetHeff(h->GetSiteOperator(isite),Lcache,Rcache);
}

void MPSImp::LoadHeffCaches(const Hamiltonian* h)
{
    CalcHeffLeft (h,itsL,true);  //This does nothing because of the 1 ???
    CalcHeffRight(h,1   ,true);
}

//
//  Used to calculate Hamiltonian L&R caches for Heff calcultions.
//
Vector3CT MPSImp::CalcHeffLeft(const MPO* o,int isite, bool cache) const
{
//    CheckSiteNumber(isite);  this function accepts out of bounds site numbers
    Vector3CT F(1,1,1,1);
    F(1,1,1)=dcmplx(1.0);
    for (int ia=1; ia<isite; ia++)
        F=itsSites[ia]->IterateLeft_F(o->GetSiteOperator(ia),F,cache);
    return F;
}
//
//  Used to calculate Hamiltonian L&R caches for Heff calcultions.
//
Vector3CT MPSImp::CalcHeffRight(const MPO* o,int isite, bool cache) const
{
//    CheckSiteNumber(isite); this function accepts out of bounds site numbers
    Vector3CT F(1,1,1,1);
    F(1,1,1)=dcmplx(1.0);
    for (int ia=itsL; ia>isite; ia--)
        F=itsSites[ia]->IterateRightF(o->GetSiteOperator(ia),F,cache);
    return F;
}


double  MPSImp::GetMaxDeltaE() const
{
    double MaxDeltaE=0.0;
    SiteLoop(ia)
    {
        double de=fabs(itsSites[ia]->GetIterDE());
        if (de>MaxDeltaE) MaxDeltaE=de;
    }
    return MaxDeltaE;
}

}; // namespace
