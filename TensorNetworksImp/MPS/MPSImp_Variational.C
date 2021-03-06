#include "TensorNetworksImp/MPS/MPSImp.H"
#include "TensorNetworksImp/MPS/Bond.H"
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
    Normalize(DRight);
    LoadHeffCaches(H);

    double DE2=0;
    for (is.begin(); !is.end(); is++)
        DE2=FindVariationalGroundState(H,*is);
    return DE2;
}

double MPSImp::FindVariationalGroundState(const Hamiltonian* H, const IterationScheduleLine& isl)
{
    assert(Logger); //Make sure we have global logger.
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
        Logger->LogInfo(2,"iter    D   dE    dlambda");
        for (; in<isl.itsMaxGSSweepIterations; in++)
        {
            Sweep(DLeft,H,isl.itsEps);  //This actually sweeps to the right, but leaves left normalized sites in its wake
            Sweep(DRight,H,isl.itsEps);
            double dE=GetMaxDeltaE();
            double dl=GetMaxDeltal(); //Max delta lambda on bonds
            Logger->LogInfoV(2,"%4d %4d %.1e %.1e",in,D,dE,dl);
            if (dE<isl.itsEps.itsDelatEnergy1Epsilon && dl<isl.itsEps.itsDeltaLambdaEpsilon) break;
        }
        in++;

        double E1=GetExpectation(H);
        double E2=GetExpectation(H2);
        DE2=E2-E1*E1;
        Logger->LogInfoV(0,"Variational GS D=%4d, %4d iterations, <E>=%.11f, <E^2>-<E>^2=%.2e",D,in,E1/(itsL-1),DE2);
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
    assert(Logger); //Make sure we have global logger.
    CheckSiteNumber(isite);
    assert(IsRLNormalized(isite));
    if (!itsSites[isite]->IsFrozen())
    {
        Logger->LogInfo(2,isite,"Calculating Heff"); //Logger will update the graphs
        Matrix6CT Heff6=GetHeffIterate(h,isite); //New iterative version
        Logger->LogInfo(2,isite,"Running eigen solver"); //Logger will update the graphs
        itsSites[isite]->Refine(Heff6.Flatten(),eps);
    }
    UpdateEnergyData(isite);
    NormalizeSite(lr,isite);
    const MPO* hmpo=h;
    itsSites[isite]->UpdateCache(hmpo->GetSiteOperator(isite),
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
    const MPO* hmpo=h;
    return itsSites[isite]->GetHeff(hmpo->GetSiteOperator(isite),Lcache,Rcache);
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


double  MPSImp::GetMaxDeltal() const
{
    double MaxDeltal=0.0;
    for (int ib=1; ib<itsL; ib++)
    {
        double dl=itsBonds[ib]->GetMaxDelta();
        if (dl>MaxDeltal) MaxDeltal=dl;
    }
    return MaxDeltal;
}

double  MPSImp::GetMaxDeltal(const BondsType& bonds_cache) const
{
    assert(bonds_cache.size()==itsBonds.size());
    double MaxDeltal=0.0;
    for (int ib=1; ib<itsL; ib++)
    {
        double dl=itsBonds[ib]->GetMaxDelta(*bonds_cache[ib]);
        if (dl>MaxDeltal) MaxDeltal=dl;
    }
    return MaxDeltal;
}


}; // namespace
