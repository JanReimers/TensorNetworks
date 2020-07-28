#include "TensorNetworksImp/MatrixProductStateImp.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/LRPSupervisor.H"
#include "Containers/Matrix4.H"
#include "Functions/Mesh/PlotableMesh.H"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

//-------------------------------------------------------------------------------
//
//  Init/construction zone
//
MatrixProductStateImp::MatrixProductStateImp(int L, double S, int D,const Epsilons& eps)
    : itsL(L)
    , itsDmax(D)
    , itsS(S)
    , itsp(2*S+1)
    , itsNSweep(0)
    , itsSelectedSite(1)
    , itsEpsilons(eps)
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
    //
    //  Create bond objects
    //
    itsBonds.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    for (int i=1;i<itsL;i++)
        itsBonds.push_back(new Bond(eps.itsSingularValueZeroEpsilon));
    //
    //  Create Sites
    //
    itsSites.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    itsSites.push_back(new MatrixProductSite(TensorNetworks::PLeft,NULL,itsBonds[1],itsp,1,D));
    for (int i=2;i<=itsL-1;i++)
        itsSites.push_back(new MatrixProductSite(TensorNetworks::PBulk,itsBonds[i-1],itsBonds[i],itsp,D,D));
    itsSites.push_back(new MatrixProductSite(TensorNetworks::PRight,itsBonds[L-1],NULL,itsp,D,1));
    //
    //  Tell each bond about its left and right sites.
    //
    for (int i=1;i<itsL;i++)
        itsBonds[i]->SetSites(itsSites[i],itsSites[i+1]);

    InitPlotting();

}

MatrixProductStateImp::~MatrixProductStateImp()
{
//    cout << "MatrixProductStateImp destructor." << endl;
    delete itsBondsPMesh;
    delete itsSitesPMesh;
    delete itsBondsMesh;
    delete itsSitesMesh;
}

#define SiteLoop(ia) for (int ia=1;ia<=itsL;ia++)
#define CheckSiteNumber(ia)\
    assert(ia>=1);\
    assert(ia<=itsL);\

#define CheckBondNumber(ib)\
    assert(ib>=1);\
    assert(ib<itsL);\



void MatrixProductStateImp::InitializeWith(TensorNetworks::State state)
{
    int sgn=1;
    SiteLoop(ia)
    {
        itsSites[ia]->InitializeWith(state,sgn);
        sgn*=-1;
    }

}

void MatrixProductStateImp::Freeze(int isite,double s)
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
void MatrixProductStateImp::Normalize(TensorNetworks::Direction LR,LRPSupervisor* Supervisor)
{
    ForLoop(LR)
        NormalizeSite(LR,ia,Supervisor);
}


void MatrixProductStateImp::NormalizeSite(TensorNetworks::Direction lr,int isite,LRPSupervisor* super)
{
    CheckSiteNumber(isite);
    std::string lrs=lr==TensorNetworks::DLeft ? "Left" : "Right";
    super->DoneOneStep(2,SiteMessage("SVD "+lrs+" Normalize site ",isite),isite);
    itsSites[isite]->SVDNormalize(lr);
    super->DoneOneStep(2,SiteMessage("SVD "+lrs+" Normalize update Bond data ",isite),isite);
    int bond_index=isite+( lr==TensorNetworks::DLeft ? 0 :-1);
    if (bond_index<itsL && bond_index>=1)
        UpdateBondData(bond_index);
}

void MatrixProductStateImp::NormalizeAndCompress(TensorNetworks::Direction LR,int      Dmax,LRPSupervisor* s)
{
    ForLoop(LR)
        NormalizeAndCompressSite(LR,ia,Dmax,0.0,s);
}

void MatrixProductStateImp::NormalizeAndCompress(TensorNetworks::Direction LR,double epsMin,LRPSupervisor* s)
{
    ForLoop(LR)
        NormalizeAndCompressSite(LR,ia,0,epsMin,s);
}

void MatrixProductStateImp::NormalizeAndCompressSite(TensorNetworks::Direction lr,int isite,int Dmax, double epsMin, LRPSupervisor* super)
{
    CheckSiteNumber(isite);
    std::string lrs=lr==TensorNetworks::DLeft ? "Left" : "Right";
    super->DoneOneStep(2,SiteMessage("SVD "+lrs+" Normalize site ",isite),isite);
    itsSites[isite]->SVDNormalize(lr,Dmax,epsMin);
    super->DoneOneStep(2,SiteMessage("SVD "+lrs+" Normalize update Bond data ",isite),isite);
    int bond_index=isite+( lr==TensorNetworks::DLeft ? 0 :-1);
    if (bond_index<itsL && bond_index>=1)
        UpdateBondData(bond_index);
}

//
//  Mixed canonical  A*A*A*A...A*M(isite)*B*B...B*B
//
void MatrixProductStateImp::Normalize(int isite,LRPSupervisor* super)
{
    CheckSiteNumber(isite);
    if (isite>1)
    {
        for (int ia=1; ia<isite; ia++)
        {
            NormalizeSite(TensorNetworks::DLeft,ia,super);
        }
//        itsSites[isite]->ReshapeFromLeft(rank);
    }

    if (isite<itsL)
    {
        for (int ia=itsL; ia>isite; ia--)
        {
            NormalizeSite(TensorNetworks::DRight,ia,super);
        }
//        itsSites[isite]->ReshapeFromRight(rank);
    }
}


void MatrixProductStateImp::UpdateBondData(int isite)
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
double MatrixProductStateImp::FindGroundState(const Hamiltonian* hamiltonian, int maxIter, const Epsilons& eps,LRPSupervisor* Supervisor)
{
    Supervisor->ReadyToStart("Right normalize");
    Normalize(TensorNetworks::DRight,Supervisor);
    Supervisor->DoneOneStep(0,"Load L&R caches");
    LoadHeffCaches(hamiltonian,Supervisor);

    double E1=0;
    for (int in=0; in<maxIter; in++)
    {
        Supervisor->DoneOneStep(0,"Sweep Right");
        E1=Sweep(TensorNetworks::DLeft ,hamiltonian,Supervisor,eps); //This actually sweeps to the right, but leaves left normalized sites in its wake
        Supervisor->DoneOneStep(0,"Sweep Left");
        E1=Sweep(TensorNetworks::DRight,hamiltonian,Supervisor,eps);
        if (GetMaxDeltaE()<eps.itsEnergyConvergenceEpsilon) break;
    }
    Supervisor->DoneOneStep(0,"Contracting <E^2>"); //Supervisor will update the graphs
    double E2=GetExpectation(hamiltonian,hamiltonian);
    return E2-E1*E1;
}

double MatrixProductStateImp::Sweep(TensorNetworks::Direction lr,const Hamiltonian* h,LRPSupervisor* Supervisor,const Epsilons& eps)
{
    int iter=0;
    ForLoop(lr)
    {
        CheckSiteNumber(ia);
        Refine(lr,h,Supervisor,eps,ia);
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
    Supervisor->DoneOneStep(0,"Calculating Expectation <E>"); //Supervisor will update the graphs
    double E1=GetExpectation(h);
    if (weHaveGraphs())
    {
        AddPoint("Iter E/J",Plotting::Point(itsNSweep,E1));
    }
    return E1;
}


void MatrixProductStateImp::Refine(TensorNetworks::Direction lr,const Hamiltonian *h,LRPSupervisor* Supervisor,const Epsilons& eps,int isite)
{
//    assert(CheckNormalized(isite,eps.itsNormalizationEpsilon));
    CheckSiteNumber(isite);
    if (!itsSites[isite]->IsFrozen())
    {
        Supervisor->DoneOneStep(2,"Calculating Heff",isite); //Supervisor will update the graphs
        Matrix6T Heff6=GetHeffIterate(h,isite); //New iterative version
        Supervisor->DoneOneStep(2,"Running eigen solver",isite); //Supervisor will update the graphs
        itsSites[isite]->Refine(Heff6.Flatten(),eps);
    }
    itsSiteEnergies[isite]=itsSites[isite]->GetSiteEnergy();
    itsSiteEGaps   [isite]=itsSites[isite]->GetEGap      ();
    NormalizeSite(lr,isite,Supervisor);
    itsSites[isite]->UpdateCache(h->GetSiteOperator(isite),
                            GetHeffCache(TensorNetworks::DLeft ,isite-1),
                            GetHeffCache(TensorNetworks::DRight,isite+1));

}

MatrixProductStateImp::Vector3T MatrixProductStateImp::GetHeffCache (TensorNetworks::Direction lr,int isite) const
{
    //CheckSiteNumber(isite); this function accepts out of range site numbers
    Vector3T H(1,1,1,1);
    H(1,1,1)=eType(1.0);
    if (isite>=1 && isite<=itsL)  H=itsSites[isite]->GetHeffCache(lr);
    return H;
}

MatrixProductStateImp::Matrix6T MatrixProductStateImp::GetHeffIterate   (const Hamiltonian* h,int isite) const
{
    CheckSiteNumber(isite);
    Vector3T Lcache=GetHeffCache(TensorNetworks::DLeft ,isite-1);
    Vector3T Rcache=GetHeffCache(TensorNetworks::DRight,isite+1);
    return itsSites[isite]->GetHeff(h->GetSiteOperator(isite),Lcache,Rcache);
}
void MatrixProductStateImp::LoadHeffCaches(const Hamiltonian* h,LRPSupervisor* supervisor)
{
    assert(supervisor);
    GetEOLeft_Iterate(h,supervisor,1,true);
    GetEORightIterate(h,supervisor,1,true);
}

double  MatrixProductStateImp::GetMaxDeltaE() const
{
    double MaxDeltaE=0.0;
    SiteLoop(ia)
    {
        double de=fabs(itsSites[ia]->GetIterDE());
        if (de>MaxDeltaE) MaxDeltaE=de;
    }
    return MaxDeltaE;
}


double   MatrixProductStateImp::GetExpectation   (const Operator* o) const
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

MatrixProductStateImp::eType   MatrixProductStateImp::GetExpectationC(const Operator* o) const
{
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    SiteLoop(ia)
        F=itsSites[ia]->IterateLeft_F(o->GetSiteOperator(ia),F);

    return F(1,1,1);
}

double   MatrixProductStateImp::GetExpectation(const Operator* o1,const Operator* o2) const
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


void  MatrixProductStateImp::ApplyInPlace(const Operator* o)
{
    SiteLoop(ia)
        itsSites[ia]->ApplyInPlace(o->GetSiteOperator(ia));
}

MatrixProductState*  MatrixProductStateImp::Apply(const Operator* o) const
{
    MatrixProductStateImp* psiPrime=new MatrixProductStateImp(itsL,itsS,1,itsEpsilons);
    SiteLoop(ia)
        itsSites[ia]->Apply(o->GetSiteOperator(ia),psiPrime->itsSites[ia]);

    return psiPrime;
}

//--------------------------------------------------------------------------------------
//
//    Reporting
//
void MatrixProductStateImp::Report(std::ostream& os) const
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

std::string MatrixProductStateImp::GetNormStatus(int isite) const
{
    CheckSiteNumber(isite);
    return itsSites[isite]->GetNormStatus(itsEpsilons.itsNormalizationEpsilon);
}

//
//  Used ot L&R caches for Heff calcultions.
//
MatrixProductStateImp::Vector3T MatrixProductStateImp::GetEOLeft_Iterate(const Operator* o,LRPSupervisor* supervisor,int isite, bool cache) const
{
//    CheckSiteNumber(isite);  this function accepts out of bounds site numbers
    assert(supervisor);
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=1; ia<isite; ia++)
    {
        supervisor->DoneOneStep(1,SiteMessage("Calculating L cache for site ",ia));
        F=itsSites[ia]->IterateLeft_F(o->GetSiteOperator(ia),F,cache);
    }
    return F;
}
//
//  Used ot L&R caches for Heff calcultions.
//
MatrixProductStateImp::Vector3T MatrixProductStateImp::GetEORightIterate(const Operator* o,LRPSupervisor* supervisor,int isite, bool cache) const
{
//    CheckSiteNumber(isite); this function accepts out of bounds site numbers
    assert(supervisor);
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=itsL; ia>isite; ia--)
    {
        supervisor->DoneOneStep(1,SiteMessage("Calculating R cache for site ",ia));
        F=itsSites[ia]->IterateRightF(o->GetSiteOperator(ia),F,cache);
    }
    return F;
}


OneSiteDMs MatrixProductStateImp::CalculateOneSiteDMs(LRPSupervisor* supervisor)
{
    OneSiteDMs ret(itsL,itsp);
    Normalize(TensorNetworks::DRight,supervisor);
    SiteLoop(ia)
    {
        supervisor->DoneOneStep(2,SiteMessage("Calculate ro(mn) site: ",ia),ia);
        ret.Insert(ia,itsSites[ia]->CalculateOneSiteDM());
        NormalizeSite(TensorNetworks::DLeft,ia,supervisor);
    }
    return ret;
}

MatrixProductStateImp::Matrix4T MatrixProductStateImp::CalculateTwoSiteDM(int ia,int ib) const
{
    CheckSiteNumber(ia);
    CheckSiteNumber(ib);
#ifdef DEBUG
    for (int is=1; is<ia; is++)
        assert(GetNormStatus(is)[0]=='A');
    for (int is=ib+1; is<=itsL; is++)
        assert(GetNormStatus(is)[0]=='B');
#endif
    Matrix4T ret(itsp,itsp,itsp,itsp,1);
    ret.Fill(eType(0.0));
    // Start the zipper
    for (int m=0; m<itsp; m++)
        for (int n=0; n<itsp; n++)
        {
            MatrixCT C=itsSites[ia]->InitializeTwoSiteDM(m,n);
            for (int ix=ia+1; ix<ib; ix++)
                C=itsSites[ix]->IterateTwoSiteDM(C);
            C=itsSites[ib]->FinializeTwoSiteDM(C);

            for (int m2=0; m2<itsp; m2++)
                for (int n2=0; n2<itsp; n2++)
                    ret(m+1,m2+1,n+1,n2+1)=C(m2+1,n2+1);
        }
    assert(IsHermitian(ret.Flatten(),1e-14));
    return ret;
}


TwoSiteDMs MatrixProductStateImp::CalculateTwoSiteDMs(LRPSupervisor* supervisor)
{
    Normalize(TensorNetworks::DRight,supervisor);
    TwoSiteDMs ret(itsL,itsp);
    SiteLoop(ia)
        for (int ib=ia+1;ib<=itsL;ib++)
        {
            Matrix4T ro=CalculateTwoSiteDM(ia,ib);
            ret.Insert(ia,ib,ro);
            NormalizeSite(TensorNetworks::DLeft,ia,supervisor);
        }
    // Normalize the last to keep things tidy
    NormalizeSite(TensorNetworks::DLeft,itsL,supervisor);
    return ret;
}
