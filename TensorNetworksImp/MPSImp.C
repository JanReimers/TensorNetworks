#include "TensorNetworksImp/MPSImp.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/TNSLogger.H"
#include "Containers/Matrix4.H"
#include "Functions/Mesh/PlotableMesh.H"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;


int GetD1(int a, int L, int d, int DMax)
{
    int D=DMax;
    if (a<=L/2)
        D=Min(static_cast<int>(pow(d,a-1)),DMax); //LHS
    else
        D=Min(static_cast<int>(pow(d,L-a+1)),DMax);  //RHS
    return D;
}

int GetD2(int a, int L, int d, int DMax)
{
    int D=DMax;
    if (a<=L/2)
        D=Min(static_cast<int>(pow(d,a)),DMax); //LHS
    else
        D=Min(static_cast<int>(pow(d,L-a)),DMax); //RHS
    return D;
}


//-------------------------------------------------------------------------------
//
//  Init/construction zone
//
MPSImp::MPSImp(int L, double S, int D,double normEps,TNSLogger* s)
    : itsL(L)
    , itsDmax(D)
    , itsS(S)
    , itsd(2*S+1)
    , itsNSweep(0)
    , itsSelectedSite(1)
    , itsNormEps(normEps)
    , itsLogger(s)
    , itsSitesMesh(0)
    , itsBondsMesh(0)
{
    assert(itsL>0);
#ifdef DEBUG
    double ipart;
    double frac=std::modf(2.0*itsS,&ipart);
    assert(frac==0.0);
#endif
    assert(itsS>=0.5);
    assert(itsDmax>0);

    if (itsLogger==0)
        itsLogger=new TNSLogger();

    InitSitesAndBonds();
    InitPlotting();

}

MPSImp::MPSImp(const MPSImp& mps)
    : itsL           (mps.itsL)
    , itsDmax        (mps.itsDmax)
    , itsS           (mps.itsS)
    , itsd           (mps.itsd)
    , itsNSweep      (mps.itsNSweep)
    , itsSelectedSite(mps.itsSelectedSite)
    , itsNormEps     (mps.itsNormEps)
    , itsLogger  (mps.itsLogger->Clone())
    , itsSitesMesh   (0)
    , itsBondsMesh   (0)
{
    InitSitesAndBonds();
    InitPlotting();
    for (int ia=1; ia<=itsL; ia++)
        itsSites[ia]->CloneState(mps.itsSites[ia]); //Transfer wave function data
    for (int ia=1; ia<itsL; ia++)
        itsBonds[ia]->CloneState(mps.itsBonds[ia]); //Transfer wave function data
}

void MPSImp::InitSitesAndBonds()
{
    //
    //  Create bond objects
    //
    itsBonds.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    for (int i=1; i<itsL; i++)
        itsBonds.push_back(new Bond());
    //
    //  Create Sites
    //
    itsSites.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    itsSites.push_back(new MPSSite(TensorNetworks::PLeft,NULL,itsBonds[1],itsd,
                       GetD1(1,itsL,itsd,itsDmax),GetD2(1,itsL,itsd,itsDmax)));
    for (int i=2; i<=itsL-1; i++)
        itsSites.push_back(new MPSSite(TensorNetworks::PBulk,itsBonds[i-1],itsBonds[i],itsd,
                           GetD1(i,itsL,itsd,itsDmax),GetD2(i,itsL,itsd,itsDmax)));
    itsSites.push_back(new MPSSite(TensorNetworks::PRight,itsBonds[itsL-1],NULL,itsd,
                       GetD1(itsL,itsL,itsd,itsDmax),GetD2(itsL,itsL,itsd,itsDmax)));
    //
    //  Tell each bond about its left and right sites.
    //
    for (int i=1; i<itsL; i++)
        itsBonds[i]->SetSites(itsSites[i],itsSites[i+1]);
}



MPSImp::~MPSImp()
{
//    cout << "MatrixProductStateImp destructor." << endl;
    delete itsBondsPMesh;
    delete itsSitesPMesh;
    delete itsBondsMesh;
    delete itsSitesMesh;
}

MPS* MPSImp::Clone() const
{
    return new MPSImp(*this);
}

#define SiteLoop(ia) for (int ia=1;ia<=itsL;ia++)
#define CheckSiteNumber(ia)\
    assert(ia>=1);\
    assert(ia<=itsL);\

#define CheckBondNumber(ib)\
    assert(ib>=1);\
    assert(ib<itsL);\



void MPSImp::InitializeWith(TensorNetworks::State state)
{
    int sgn=1;
    SiteLoop(ia)
    {
        itsSites[ia]->InitializeWith(state,sgn);
        sgn*=-1;
    }

}

void MPSImp::Freeze(int isite,double s)
{
    CheckSiteNumber(isite);
    itsSites[isite]->Freeze(s);
}

c_str SiteMessage(const std::string& message,int isite)
{
    static std::string ret;
    ret=message+std::to_string(isite);
    return ret.c_str();
}

#define ForLoop(LR) for (int ia=GetStart(LR);ia!=GetStop(LR);ia+=GetIncrement(LR))
//-------------------------------------------------------------------------------
//
//   Normalization routines
//
void MPSImp::Normalize(TensorNetworks::Direction LR)
{
    ForLoop(LR)
    NormalizeSite(LR,ia);
}


void MPSImp::NormalizeSite(TensorNetworks::Direction lr,int isite)
{
    CheckSiteNumber(isite);
    std::string lrs=lr==TensorNetworks::DLeft ? "Left" : "Right";
    itsLogger->DoneOneStep(2,SiteMessage("SVD "+lrs+" Normalize site ",isite),isite);
//    cout << "SVD " << lrs << " site " << isite << endl;
    itsSites[isite]->SVDNormalize(lr);
    itsLogger->DoneOneStep(2,SiteMessage("SVD "+lrs+" Normalize update Bond data ",isite),isite);
    int bond_index=isite+( lr==TensorNetworks::DLeft ? 0 :-1);
    if (bond_index<itsL && bond_index>=1)
        UpdateBondData(bond_index);
}

void MPSImp::NormalizeAndCompress(TensorNetworks::Direction LR,int      Dmax)
{
//    SetCanonicalBondDimensions(Invert(LR)); //Sweep backwards and set proper bond dimensions
    ForLoop(LR)
    NormalizeAndCompressSite(LR,ia,Dmax,0.0);
    itsDmax=Dmax;
}

void MPSImp::NormalizeAndCompress(TensorNetworks::Direction LR,double epsMin)
{
    ForLoop(LR)
    NormalizeAndCompressSite(LR,ia,0,epsMin);
    assert(false); //We need a way to set Dmax
//    itsDmax=Dmax;
//    SetCanonicalBondDimensions(Invert(LR)); //Sweep backwards and set proper bond dimensions
}

void MPSImp::NormalizeAndCompressSite(TensorNetworks::Direction lr,int isite,int Dmax, double epsMin)
{
    CheckSiteNumber(isite);
//    cout << "----- Normalize and Compress site " << isite << " " << GetNormStatus() << " -----" << endl;
    std::string lrs=lr==TensorNetworks::DLeft ? "Left" : "Right";
    itsLogger->DoneOneStep(2,SiteMessage("SVD "+lrs+" Normalize site ",isite),isite);
    //cout << "SVD " << lrs << " site " << isite << endl;
    itsSites[isite]->SVDNormalize(lr,Dmax,epsMin);

    itsLogger->DoneOneStep(2,SiteMessage("SVD "+lrs+" Normalize update Bond data ",isite),isite);
    int bond_index=isite+( lr==TensorNetworks::DLeft ? 0 :-1);
    if (bond_index<itsL && bond_index>=1)
        UpdateBondData(bond_index);
}

//
//  Mixed canonical  A*A*A*A...A*M(isite)*B*B...B*B
//
void MPSImp::Normalize(int isite)
{
    CheckSiteNumber(isite);
    if (isite>1)
    {
        for (int ia=1; ia<isite; ia++)
        {
            NormalizeSite(TensorNetworks::DLeft,ia);
        }
//        itsSites[isite]->ReshapeFromLeft(rank);
    }

    if (isite<itsL)
    {
        for (int ia=itsL; ia>isite; ia--)
        {
            NormalizeSite(TensorNetworks::DRight,ia);
        }
//        itsSites[isite]->ReshapeFromRight(rank);
    }
}

void MPSImp::SetCanonicalBondDimensions(TensorNetworks::Direction LR)
{
    assert(false); //Make sure we are not using this right now.
    int D1= LR==TensorNetworks::DLeft ? 1    : itsd;
    int D2= LR==TensorNetworks::DLeft ? itsd : 1   ;

    ForLoop(LR)
    {
        itsSites[ia]->SetCanonicalBondDimensions(D1,D2);
        if (D1>itsDmax && D2>itsDmax) break;
        D1*=itsd;
        D2*=itsd;
    }
}

void MPSImp::UpdateBondData(int isite)
{
    CheckBondNumber(isite);
    itsBondEntropies[isite]=itsBonds[isite]->GetBondEntropy();
    itsBondMinSVs   [isite]=log10(itsBonds[isite]->GetMinSV());
    itsBondRanks    [isite]=itsBonds[isite]->GetRank();
    if (isite==itsSelectedSite)
        itssSelectedEntropySpectrum=itsBonds[isite]->GetSVs();
}

//--------------------------------------------------------------------------------------
//
// Find ground state
//
double MPSImp::FindVariationalGroundState(const Hamiltonian* H,const IterationSchedule& is)
{
    double dE=0;
    for (is.begin(); !is.end(); is++)
        dE=FindVariationalGroundState(H,*is);
    return dE;
}

double MPSImp::FindVariationalGroundState(const Hamiltonian* hamiltonian, const IterationScheduleLine& isl)
{
    itsLogger->ReadyToStart("Right normalize");
    Normalize(TensorNetworks::DRight);
    itsLogger->DoneOneStep(0,"Load L&R caches");
    LoadHeffCaches(hamiltonian);

    double E1=0;
    for (int in=0; in<isl.itsMaxGSSweepIterations; in++)
    {
        itsLogger->DoneOneStep(0,"Sweep Right");
        E1=Sweep(TensorNetworks::DLeft,hamiltonian,isl.itsEps);  //This actually sweeps to the right, but leaves left normalized sites in its wake
        itsLogger->DoneOneStep(0,"Sweep Left");
        E1=Sweep(TensorNetworks::DRight,hamiltonian,isl.itsEps);
        if (GetMaxDeltaE()<isl.itsEps.itsDelatEnergy1Epsilon) break;
    }
    itsLogger->DoneOneStep(0,"Contracting <E^2>"); //Logger will update the graphs
    double E2=GetExpectation(hamiltonian,hamiltonian);
    return E2-E1*E1;
}

double MPSImp::Sweep(TensorNetworks::Direction lr,const Hamiltonian* h,const Epsilons& eps)
{
    int iter=0;
    ForLoop(lr)
    {
        CheckSiteNumber(ia);
        Refine(lr,h,eps,ia);
        if (weHaveGraphs())
        {
            double de=fabs(itsSites[ia]->GetIterDE());
            //cout << "ia,diter,de=" << ia << " " << diter << " " << de << endl;
            if (de<1e-16) de=1e-16;
            double diter=itsNSweep+static_cast<double>(iter)/(itsL-1); //Fractional iter count for log(dE) plot
            AddPoint("Iter log(dE/J)",Plotting::Point(diter,log10(de)));
        }
        iter++; //ia doesn;t always count upwards, but this guy does.
    }
    itsNSweep++;
    itsLogger->DoneOneStep(0,"Calculating Expectation <E>"); //Logger will update the graphs
    double E1=GetExpectation(h);
    if (weHaveGraphs())
    {
        AddPoint("Iter E/J",Plotting::Point(itsNSweep,E1));
    }
    return E1;
}


void MPSImp::Refine(TensorNetworks::Direction lr,const Hamiltonian *h,const Epsilons& eps,int isite)
{
//    assert(CheckNormalized(isite,eps.itsNormalizationEpsilon));
    CheckSiteNumber(isite);
    assert(IsRLNormalized(isite));
    if (!itsSites[isite]->IsFrozen())
    {
        itsLogger->DoneOneStep(2,"Calculating Heff",isite); //Logger will update the graphs
        Matrix6T Heff6=GetHeffIterate(h,isite); //New iterative version
        itsLogger->DoneOneStep(2,"Running eigen solver",isite); //Logger will update the graphs
        itsSites[isite]->Refine(Heff6.Flatten(),eps);
    }
    itsSiteEnergies[isite]=itsSites[isite]->GetSiteEnergy();
    itsSiteEGaps   [isite]=itsSites[isite]->GetEGap      ();
    NormalizeSite(lr,isite);
    itsSites[isite]->UpdateCache(h->GetSiteOperator(isite),
                                 GetHeffCache(TensorNetworks::DLeft,isite-1),
                                 GetHeffCache(TensorNetworks::DRight,isite+1));

}

MPSImp::Vector3T MPSImp::GetHeffCache (TensorNetworks::Direction lr,int isite) const
{
    //CheckSiteNumber(isite); this function accepts out of range site numbers
    Vector3T H(1,1,1,1);
    H(1,1,1)=eType(1.0);
    if (isite>=1 && isite<=itsL)  H=itsSites[isite]->GetHeffCache(lr);
    return H;
}


MPSImp::Matrix6T MPSImp::GetHeffIterate   (const Hamiltonian* h,int isite) const
{
    CheckSiteNumber(isite);
    Vector3T Lcache=GetHeffCache(TensorNetworks::DLeft,isite-1);
    Vector3T Rcache=GetHeffCache(TensorNetworks::DRight,isite+1);
    return itsSites[isite]->GetHeff(h->GetSiteOperator(isite),Lcache,Rcache);
}

void MPSImp::LoadHeffCaches(const Hamiltonian* h)
{
    CalcHeffLeft (h,itsL,true);  //This does nothing because of the 1 ???
    CalcHeffRight(h,1   ,true);
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

double MPSImp::FindiTimeGroundState(const Hamiltonian* H,const IterationSchedule& is)
{
    double dE=0;
    for (is.begin(); !is.end(); is++)
        dE=FindiTimeGroundState(H,*is);
    return dE;
}

double MPSImp::FindiTimeGroundState(const Hamiltonian* H,const IterationScheduleLine& isl)
{
    double E1=GetExpectation(H)/(itsL-1);
    cout << isl << endl;
    cout.precision(5);
    cout << "E=" << std::fixed << E1 << endl;
    MPO* W =H->CreateOperator(isl.itsdt,isl.itsTrotterOrder);
    double percent=W->Compress(0,isl.itsEps.itsMPOCompressEpsilon);
    cout << "FindGroundState dt=" << isl.itsdt << " " << percent << "% compresstion" << endl;
    for (int niter=1; niter<isl.itsMaxGSSweepIterations; niter++)
    {
        ApplyInPlace(W); //this now has large D_2 = D_1*Dw
        MPS* Psi2=Clone(); //Make copy of the uncompressed Psi
        //
        //  Compress in both directions
        //
        NormalizeAndCompress(TensorNetworks::DLeft ,isl.itsDmax);
        NormalizeAndCompress(TensorNetworks::DRight,isl.itsDmax);
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
        cout << "n=" << niter << "  E=" << std::fixed << Enew << std::scientific << "  dE=" << dE << endl;
        E1=Enew;
        if (fabs(dE)<=isl.itsEps.itsDelatEnergy1Epsilon) break;
    }
    double E2=GetExpectation(H,H);
    return E2-E1*E1;
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
        itsLogger->DoneOneStep(0,"Sweep Right");
        double O1=Sweep(TensorNetworks::DLeft,Psi2);  //This actually sweeps to the right, but leaves left normalized sites in its wake

//        cout << "Left  " << in << " Norm error=" << O1 << endl;
        itsLogger->DoneOneStep(0,"Sweep Left");
        double O2=Sweep(TensorNetworks::DRight,Psi2);
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

double MPSImp::Sweep(TensorNetworks::Direction lr,const MPS* Psi2)
{
    int iter=0;
    const MPSImp* psi2Imp=dynamic_cast<const MPSImp*>(Psi2);
    assert(psi2Imp);
    MatrixCT MTrace(1,1);
    MTrace(1,1)=eType(1.0);
    ForLoop(lr)
    {
        CheckSiteNumber(ia);
//        cout << "----- Opimizing site " << ia << " " << GetNormStatus() << " -----" << endl;

        assert(IsRLNormalized(ia));
        assert(GetRLCache(TensorNetworks::DLeft ,ia-1)==CalcHeffLeft(Psi2,ia,false));
        assert(GetRLCache(TensorNetworks::DRight,ia+1)==CalcHeffRight(Psi2,ia,false));
        itsSites[ia]->Optimize(psi2Imp->itsSites[ia],
                               GetRLCache(TensorNetworks::DLeft,ia-1),
                               GetRLCache(TensorNetworks::DRight,ia+1));
        MTrace=itsSites[ia]->IterateF(lr,MTrace);
        NormalizeSite(lr,ia);
        itsSites[ia]->UpdateCache(psi2Imp->itsSites[ia],
                                  GetRLCache(TensorNetworks::DLeft,ia-1),
                                  GetRLCache(TensorNetworks::DRight,ia+1));

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

MPSImp::MatrixCT MPSImp::GetRLCache (TensorNetworks::Direction lr,int isite) const
{
    //CheckSiteNumber(isite); this function accepts out of range site numbers
    MatrixCT RL(1,1);
    RL(1,1)=eType(1.0);
    if (isite>=1 && isite<=itsL)  RL=itsSites[isite]->GetRLCache(lr);
    return RL;
}

void MPSImp::LoadCaches(const MPS* Psi2)
{
    CalcHeffLeft(Psi2,itsL,true);
    CalcHeffRight(Psi2,   1,true);
}



double   MPSImp::GetOverlap(const MPS* Psi2) const
{
    const MPSImp* Psi2Imp=dynamic_cast<const MPSImp*>(Psi2);
    assert(Psi2Imp);

    MatrixCT F(1,1);
    F(1,1)=eType(1.0);
    SiteLoop(ia)
    F=itsSites[ia]->IterateLeft_F(Psi2Imp->itsSites[ia],F);

    assert(F.GetLimits()==MatLimits(1,1));
    double iO=std::imag(F(1,1));
    if (fabs(iO)>1e-10)
        cout << "Warning: MatrixProductState::GetOverlap Imag(O)=" << std::imag(F(1,1)) << endl;

    return std::real(F(1,1));
}

double   MPSImp::GetExpectation   (const Operator* o) const
{
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    SiteLoop(ia)
    F=itsSites[ia]->IterateLeft_F(o->GetSiteOperator(ia),F);

    double iE=std::imag(F(1,1,1));
    if (fabs(iE)>1e-10)
        cout << "Warning: MatrixProductState::GetExpectation Imag(E)=" << std::imag(F(1,1,1)) << endl;

    return std::real(F(1,1,1));
}

MPSImp::eType   MPSImp::GetExpectationC(const Operator* o) const
{
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    SiteLoop(ia)
    F=itsSites[ia]->IterateLeft_F(o->GetSiteOperator(ia),F);

    return F(1,1,1);
}

double   MPSImp::GetExpectation(const Operator* o1,const Operator* o2) const
{
    Vector4T F(1,1,1,1,1);
    F(1,1,1,1)=eType(1.0);
    SiteLoop(ia)
    F=itsSites[ia]->IterateLeft_F(o1->GetSiteOperator(ia),o2->GetSiteOperator(ia),F);

    double iE=std::imag(F(1,1,1,1));
    if (fabs(iE)>1e-10)
        cout << "Warning: MatrixProductState::GetExpectation Imag(E)=" << std::imag(F(1,1,1,1)) << endl;

    return std::real(F(1,1,1,1));
}


void  MPSImp::ApplyInPlace(const Operator* o)
{
    SiteLoop(ia)
    itsSites[ia]->ApplyInPlace(o->GetSiteOperator(ia));
}

MPS*  MPSImp::Apply(const Operator* o) const
{
    MPSImp* psiPrime=new MPSImp(itsL,itsS,1,itsNormEps,itsLogger->Clone());
    SiteLoop(ia)
    {
//        cout << "------------- Site " << ia << "-----------------" << endl;
        itsSites[ia]->Apply(o->GetSiteOperator(ia),psiPrime->itsSites[ia]);
    }

    return psiPrime;
}

//--------------------------------------------------------------------------------------
//
//    Reporting
//
void MPSImp::Report(std::ostream& os) const
{
    os.precision(3);
    os << "Matrix product state for " << itsL << " lattice sites." << endl;
    os << "  Site  D1  D2  Bond Entropy   #updates  Rank  Sparsisty     Emin      Egap    dA" << endl;
    SiteLoop(ia)
    {
        os << std::setw(3) << ia << "  ";
        itsSites[ia]->Report(os);
        os << endl;
    }
}

char MPSImp::GetNormStatus(int isite) const
{
    CheckSiteNumber(isite);
    return itsSites[isite]->GetNormStatus(itsNormEps);
}

std::string MPSImp::GetNormStatus() const
{
    std::string ret(itsL,' ');
    SiteLoop(ia)
    {
        ret[ia-1]=itsSites[ia]->GetNormStatus(itsNormEps);
    }
    return ret;
}


bool MPSImp::IsRLNormalized(int isite) const
{
    bool ret=true;
    for (int ia=1; ia<isite; ia++)
        ret=ret&&itsSites[ia]->GetNormStatus(itsNormEps)=='A';
    for (int ia=isite+1; ia<=itsL; ia++)
        ret=ret&&itsSites[ia]->GetNormStatus(itsNormEps)=='B';

    if (!ret) cout << "Unormailzed state " << GetNormStatus() << endl;
    return ret;
}
//
//  Used to calculate Hamiltonian L&R caches for Heff calcultions.
//
MPSImp::Vector3T MPSImp::CalcHeffLeft(const Operator* o,int isite, bool cache) const
{
//    CheckSiteNumber(isite);  this function accepts out of bounds site numbers
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=1; ia<isite; ia++)
    {
        itsLogger->DoneOneStep(1,SiteMessage("Calculating L cache for site ",ia));
        F=itsSites[ia]->IterateLeft_F(o->GetSiteOperator(ia),F,cache);
    }
    return F;
}
//
//  Used to calculate Hamiltonian L&R caches for Heff calcultions.
//
MPSImp::Vector3T MPSImp::CalcHeffRight(const Operator* o,int isite, bool cache) const
{
//    CheckSiteNumber(isite); this function accepts out of bounds site numbers
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=itsL; ia>isite; ia--)
    {
        itsLogger->DoneOneStep(1,SiteMessage("Calculating R cache for site ",ia));
        F=itsSites[ia]->IterateRightF(o->GetSiteOperator(ia),F,cache);
    }
    return F;
}
//
//  Used to calculate  L&R caches for <psi_tilde|psi> calcultions.
//
MPSImp::MatrixCT MPSImp::CalcHeffLeft(const MPS* psi2,int isite, bool cache) const
{
//    CheckSiteNumber(isite);  this function accepts out of bounds site numbers
    const MPSImp* psi2Imp=dynamic_cast<const MPSImp*>(psi2);
    assert(psi2Imp);

    MatrixCT F(1,1);
    F(1,1)=eType(1.0);
    for (int ia=1; ia<isite; ia++)
    {
        itsLogger->DoneOneStep(1,SiteMessage("Calculating L cache for site ",ia));
        F=itsSites[ia]->IterateLeft_F(psi2Imp->itsSites[ia],F,cache);
    }
    return F;
}
//
//  Used to calculate  L&R caches for <psi_tilde|psi> calcultions.
//
MPSImp::MatrixCT MPSImp::CalcHeffRight(const MPS* psi2,int isite, bool cache) const
{
//    CheckSiteNumber(isite); this function accepts out of bounds site numbers
    const MPSImp* psi2Imp=dynamic_cast<const MPSImp*>(psi2);
    assert(psi2Imp);
    MatrixCT F(1,1);
    F(1,1)=eType(1.0);
    for (int ia=itsL; ia>isite; ia--)
    {
        itsLogger->DoneOneStep(1,SiteMessage("Calculating R cache for site ",ia));
        F=itsSites[ia]->IterateRightF(psi2Imp->itsSites[ia],F,cache);
    }
    return F;
}


OneSiteDMs MPSImp::CalculateOneSiteDMs()
{
    OneSiteDMs ret(itsL,itsd);
    Normalize(TensorNetworks::DRight);
    SiteLoop(ia)
    {
        itsLogger->DoneOneStep(2,SiteMessage("Calculate ro(mn) site: ",ia),ia);
        ret.Insert(ia,itsSites[ia]->CalculateOneSiteDM());
        NormalizeSite(TensorNetworks::DLeft,ia);
    }
    return ret;
}

MPSImp::Matrix4T MPSImp::CalculateTwoSiteDM(int ia,int ib) const
{
    CheckSiteNumber(ia);
    CheckSiteNumber(ib);
#ifdef DEBUG
    for (int is=1; is<ia; is++)
        assert(GetNormStatus(is)=='A');
    for (int is=ib+1; is<=itsL; is++)
        assert(GetNormStatus(is)=='B');
#endif
    Matrix4T ret(itsd,itsd,itsd,itsd,1);
    ret.Fill(eType(0.0));
    // Start the zipper
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            MatrixCT C=itsSites[ia]->InitializeTwoSiteDM(m,n);
            for (int ix=ia+1; ix<ib; ix++)
                C=itsSites[ix]->IterateTwoSiteDM(C);
            C=itsSites[ib]->FinializeTwoSiteDM(C);

            for (int m2=0; m2<itsd; m2++)
                for (int n2=0; n2<itsd; n2++)
                    ret(m+1,m2+1,n+1,n2+1)=C(m2+1,n2+1);
        }
    assert(IsHermitian(ret.Flatten(),1e-14));
    return ret;
}


TwoSiteDMs MPSImp::CalculateTwoSiteDMs()
{
    Normalize(TensorNetworks::DRight);
    TwoSiteDMs ret(itsL,itsd);
    SiteLoop(ia)
    for (int ib=ia+1; ib<=itsL; ib++)
    {
        Matrix4T ro=CalculateTwoSiteDM(ia,ib);
        ret.Insert(ia,ib,ro);
        NormalizeSite(TensorNetworks::DLeft,ia);
    }
    // Normalize the last to keep things tidy
    NormalizeSite(TensorNetworks::DLeft,itsL);
    return ret;
}
