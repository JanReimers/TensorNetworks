#include "TensorNetworksImp/iMPS/iMPSImp.H"
#include "TensorNetworksImp/MPS/Bond.H"
#include "TensorNetworks/iHamiltonian.H"
#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/TNSLogger.H"
#include "TensorNetworks/CheckSpin.H"
#include "TensorNetworks/TNSLogger.H"
#include "Operators/OperatorValuedMatrix.H"
#include "Containers/Matrix4.H"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

namespace TensorNetworks
{

//-------------------------------------------------------------------------------
//
//  Init/construction zone
//
iMPSImp::iMPSImp(int L, double S, int D,double normEps,double epsSV)
    : itsL(L)
    , itsS(S)
    , itsd(2*S+1)
    , itsNormEps(normEps)
{
    assert(itsL>0);
    assert(isValidSpin(S));
    assert(itsS>=0.5);
    assert(D>0);
    assert(Logger); //Make sure we have global logger.

    InitSitesAndBonds(D,epsSV);
}

void iMPSImp::InitSitesAndBonds(int D,double epsSV)
{
    //
    //  Create bond objects
    //
    itsBonds.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    for (int i=1; i<=itsL; i++)
        itsBonds.push_back(new Bond(D,epsSV));
    //
    //  Create Sites
    //
    itsSites.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    itsSites.push_back(new iMPSSite(itsBonds[itsL],itsBonds[1],itsd,D,1));
    for (int i=2; i<=itsL; i++)
        itsSites.push_back(new iMPSSite(itsBonds[i-1],itsBonds[i],itsd,D,i));
    //
    //  Tell each bond about its left and right sites.
    //
    for (int i=1; i<itsL; i++)
        itsBonds[i]->SetSites(itsSites[i],itsSites[i+1]);
    itsBonds[itsL]->SetSites(itsSites[itsL],itsSites[1]);
}



iMPSImp::~iMPSImp()
{
}


void iMPSImp::InitializeWith(State state)
{
    int sgn=1;
    SiteLoop(ia)
    {
        itsSites[ia]->InitializeWith(state,sgn);
        sgn*=-1;
    }

}

const iMPSSite* iMPSImp::GetSite(Direction lr,int ia) const
{
    return const_cast<iMPSImp*>(this)->GetSite(lr,ia);
}

iMPSSite* iMPSImp::GetSite(Direction lr,int ia)
{
    switch (lr)
    {
    case DLeft:
        ia--;
        break;
    case DRight:
        ia++;
        break;
    }
    return GetSite(ia);
}

const iMPSSite* iMPSImp::GetSite(int ia) const
{
    return const_cast<iMPSImp*>(this)->GetSite(ia);
}

iMPSSite* iMPSImp::GetSite(int ia)
{
    int i=(ia-1+itsL)%itsL+1;
    assert(i>=1);
    assert(i<=itsL);
    return itsSites[i];
}

//void MPSImp::IncreaseBondDimensions(int D)
//{
//    int Dideal=1;
//    int ia=1;
//    for (; ia<=itsL/2; ia++)
//    {
//        int D1=Min(Dideal,D);
//        int D2=Min(Dideal*itsd,D);
//        itsSites[ia]->NewBondDimensions(D1,D2,true);
//        assert(ia<itsL);
//        itsBonds[ia]->NewBondDimension(D2);
//        Dideal*=itsd;
//    }
//    Dideal=1;
//    for (int ib=itsL; ib>=ia; ib--)
//    {
//        int D2=Min(Dideal,D);
//        int D1=Min(Dideal*itsd,D);
//        itsSites[ib]->NewBondDimensions(D1,D2,true); //Very important to save data
//        assert(ib>1);
//        itsBonds[ib-1]->NewBondDimension(D1);
//        Dideal*=itsd;
//    }
//
//}


//--------------------------------------------------------------------------------------
//
//    Reporting
//
int iMPSImp::GetMaxD() const
{
    int D=0;
    SiteLoop(ia) D=Max(D,itsSites[ia]->GetD2());
    return D;
}

void iMPSImp::Report(std::ostream& os) const
{
    os << "Matrix product state for " << itsL << " lattice sites." << endl;
    os << "  Site  D1  D2  Norm #updates  Emin        Egap     dE" << endl;
    SiteLoop(ia)
    {
        os << std::setw(3) << ia << "  ";
        itsSites[ia]->Report(os);
        os << endl;
    }
    os << "  Bond  D   Rank  Entropy   Min(Sv)   SvError " << endl;
    for (int ib=1; ib<=itsL; ib++)
    {
        os << std::setw(3) << ib << "  ";
        itsBonds[ib]->Report(os);
        os << endl;
    }
}

char iMPSImp::GetNormStatus(int isite) const
{
    CheckSiteNumber(isite);
    return itsSites[isite]->GetNormStatus(itsNormEps);
}

std::string iMPSImp::GetNormStatus() const
{
    std::string ret(itsL,' ');
    SiteLoop(ia)
    {
        ret[ia-1]=itsSites[ia]->GetNormStatus(itsNormEps);
    }
    return ret;
}


bool iMPSImp::IsRLNormalized(int isite) const
{
    bool ret=true;
    for (int ia=1; ia<isite; ia++)
    {
        char n=itsSites[ia]->GetNormStatus(itsNormEps);
        ret=ret&& (n=='A' || n=='I');
    }
    for (int ia=isite+1; ia<=itsL; ia++)
    {
        char n=itsSites[ia]->GetNormStatus(itsNormEps);
        ret=ret&& (n=='B' || n=='I');
    }

    if (!ret) cout << "Unormalized state " << GetNormStatus() << endl;
    return ret;
}

void iMPSImp::NormalizeSvD(Direction LR)
{

}

template<typename T>
class reverse {
private:
  T& iterable_;
public:
  explicit reverse(T& iterable) : iterable_{iterable} {}
  auto begin() const { return std::rbegin(iterable_); }
  auto end() const { return std::rend(iterable_); }
};


void iMPSImp::NormalizeQR (Direction lr,double eps)
{
    int L=GetL();
    for (int ia=1;ia<=L;ia++)
        itsSites[ia]->InitQRIter(); //Reset all G's to unit

    double eta=0.0;
    int niter=0,maxIter=500;
    do
    {
        eta=0.0;
        switch (lr)
        {
        case DLeft:
            for (int ia=1;ia<=L;ia++)
                eta=Max(eta,itsSites[ia]->QRStep(lr,eps));
            break;
        case DRight:
            for (int ia=L;ia>=1;ia--)
                eta=Max(eta,itsSites[ia]->QRStep(lr,eps));
            break;
        }
        niter++;
    } while (eta>eps && niter <maxIter);
    if (niter==maxIter)
        std::cout << "iMPSImp::NormalizeQR failed to converge, eta=" << eta << std::endl;

    for (int ia=1;ia<=itsL;ia++)
        itsSites[ia]->SaveAB_CalcLR(lr); //  for each site Save A=M, and calc the left dominant eigen vector of the left transfer matrix

}

void iMPSImp::Normalize(double eps)
{
    NormalizeQR(DLeft,eps);

    NormalizeQR(DRight,eps);
}
//
//  Check ||A(k)*G(k)-G(k-1)B(k)||
//
double iMPSImp::GetGaugeError() const
{
    double etaG=0.0;
    for (int ia=1;ia<=itsL;ia++)
        etaG+=itsSites[ia]->GetGaugeError(GetSite(DLeft,ia));
    return etaG/itsL;
}


//double   iMPSImp::GetExpectation (const iHamiltonian* H) const
//{
//    assert(o);
//    double E=0.0;
//    iMPO* Hcell=H->MakeUnitcelliMPO(itsL); //Product of Ws for the whole unit cell
//    MatrixOR Wcell=Hcell->GetSiteOperator(1)->GetW();
//    for (int ia=1;ia<=itsL;ia++)
//        E+=itsSites[ia]->GetExpectation(Wcell,MakeUnitcelliMPS(ia));
//    return E/itsL;
//}

double   iMPSImp::GetExpectation (const iMPO* o) const
{
    assert(o);
    double E=0.0;
    MatrixOR W=o->GetSiteOperator(1)->GetW();
    const iHamiltonian* iH=dynamic_cast<const iHamiltonian*>(o);
    if (iH)
    {
        iMPO* Hcell=iH->MakeUnitcelliMPO(itsL); //Product of Ws for the whole unit cell
        W=Hcell->GetSiteOperator(1)->GetW();
    }
    for (int ia=1;ia<=itsL;ia++)
        E+=itsSites[ia]->GetExpectation(W,MakeUnitcelliMPS(ia))/itsL;
    return E/itsL;
}

double iMPSImp::FindVariationalGroundState(const iHamiltonian* H,const IterationSchedule& is)
{
    double E=0.0;
    Normalize(1e-14); //Sweep left and right storing A's and B's.
    iMPO* Hcell=H->MakeUnitcelliMPO(itsL); //Product of Ws for the whole unit cell
    for (is.begin(); !is.end(); is++)
        E=FindVariationalGroundState(H,Hcell,*is);
    return E;
}

double iMPSImp::FindVariationalGroundState(const iHamiltonian* H,const iMPO* Hcell,const IterationScheduleLine& isl)
{
    assert(Logger); //Make sure we have global logger.
    Logger->LogInfo(2,"iter      E            dE     :        Energies             :   Eigen values        :                     eta                         : sing. Values");
    Logger->LogInfo(2,"                              :   eL    eR     eLp    eRm   :   eM     eGR    eGL   :   L        R       Ls       Rs      G1      G2  : dL   dR  minL   minR");

    MatrixOR Wcell=Hcell->GetSiteOperator(1)->GetW();
    MatrixOR W    =H    ->GetSiteOperator(1)->GetW();
    int in=0;
    double eta=1.0;
    double dE=0.0,E=0.0;
    RefineData rd;
    for (; in<isl.itsMaxGSSweepIterations; in++)
    {
        dE=E=0.0;
        if (itsL==1)
        {
            eta=itsSites[1]->RefineOneSite(W,isl.itsEps);
            E+= itsSites[1]->GetSiteEnergy();
            dE=std::max(dE,fabs(itsSites[1]->GetIterDE()));
        }
        else for (int k=1;k<=itsL;k++)
        {
            iMPSSite* s =GetSite(k  );
            iMPSSite* sm=GetSite(k-1);
            rd=s->Refine(W,Wcell,sm,MakeUnitcelliMPS(k),isl.itsEps);
            E+= s->GetSiteEnergy()/itsL;
            dE=std::max(dE,fabs(s->GetIterDE()));
//            cout << "Site k   etaG,etaA,etaB=" << s ->GetGaugeError(sm) << " " << s ->GetANormError() << " " << s ->GetBNormError() << endl;
//            cout << "Site k-1 etaG,etaA,etaB=" << sm->GetGaugeError(s ) << " " << sm->GetANormError() << " " << sm->GetBNormError() << endl;
            //Normalize(1e-14); //Sweep left and right storing A's and B's.
            Logger->LogInfoV(2,"%4d %16.13f %.1e : %6.3f %6.3f %6.3f %6.3f : %6.3f %6.3f  %6.3f : %.1e %.1e %.1e %.1e %.1e %.1e : %.1e %.1e %.1e %.1e"
                             ,in,E/itsL,dE,
                             rd.eL,rd.eR,rd.eLp,rd.eRm,rd.eM,rd.eGR,rd.eGL,
                             rd.etaL,rd.etaR,rd.etaLs,rd.etaRs,rd.etaG1,rd.etaG2,
                             rd.dsL,rd.dsR,rd.minsL,rd.minsR);
        }
        E/=itsL;
        if (eta<isl.itsEps.itsDeltaLambdaEpsilon && dE <isl.itsEps.itsDelatEnergy1Epsilon) break;
    }
    Logger->LogInfoV(0,"Variational iMPS GS D=%4d, %4d iterations, <E>=%.13f",GetMaxD(),in,E);
    return E;
}

//
//  We actually need 2 version eachs of Acell,Bcell
//      For H_M(k) (H_AC) we need Acell(k-1) and Bcell(k+1)
//      For H_G(k) (H_C)  we need Acell(k) and Bcell(k+1)
//      For H_G(k-1)      we need Acell(k-1) and Bcell(k)
//

iMPSImp::UnitcellMPSType iMPSImp::MakeUnitcelliMPS(int k) const
{
    int ka=k-1,kb=k+1;
    Tensor3 A =GetSite(ka  )->itsA;
    Tensor3 Ap=GetSite(ka+1)->itsA;
    Tensor3 B =GetSite(kb  )->itsB;
    Tensor3 Bm=GetSite(kb-1)->itsB;
    for (int ia=1;ia<itsL;ia++)
    {
        ka--;
        kb++;
        A =HorizontalProduct(GetSite(ka  )->itsA,A );
        Ap=HorizontalProduct(GetSite(ka+1)->itsA,Ap);
        B =HorizontalProduct(B ,GetSite(kb  )->itsB);
        Bm=HorizontalProduct(Bm,GetSite(kb-1)->itsB);
    }
//    cout << "MakeUnitcelliMPS A,Ap,B,Bm=" << A.GetLabel() << " " <<  Ap.GetLabel() << " " <<  B.GetLabel() << " " <<  Bm.GetLabel() << endl;
    return std::make_tuple(A,Ap,B,Bm);
}

}; // namespace
