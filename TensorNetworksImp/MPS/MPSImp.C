#include "TensorNetworksImp/MPS/MPSImp.H"
#include "TensorNetworksImp/MPS/Bond.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/TNSLogger.H"
#include "TensorNetworks/CheckSpin.H"
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
MPSImp::MPSImp(int L, double S, int D,double normEps,double epsSV)
    : itsL(L)
    , itsS(S)
    , itsd(2*S+1)
    , itsNSweep(0)
    , itsNormEps(normEps)
{
    assert(itsL>0);
    assert(isValidSpin(S));
    assert(itsS>=0.5);
    assert(D>0);
    assert(Logger); //Make sure we have global logger.

    InitSitesAndBonds(D,epsSV);
}

MPSImp::MPSImp(const MPSImp& mps)
    : itsL           (mps.itsL)
    , itsS           (mps.itsS)
    , itsd           (mps.itsd)
    , itsNSweep      (mps.itsNSweep)
    , itsNormEps     (mps.itsNormEps)
{
    assert(Logger); //Make sure we have global logger.
    int D=mps.GetMaxD();
    assert(D>0);
    InitSitesAndBonds(D,0); //Clone state should transfer ,double epsSV
    for (int ia=1; ia<=itsL; ia++)
        itsSites[ia]->CloneState(mps.itsSites[ia]); //Transfer wave function data
    for (int ia=1; ia<itsL; ia++)
        itsBonds[ia]->CloneState(mps.itsBonds[ia]); //Transfer wave function data
}

MPSImp::MPSImp(int L, double S,Direction lr,double normEps)
    : itsL(L)
    , itsS(S)
    , itsd(2*S+1)
    , itsNSweep(0)
    , itsNormEps(normEps)
{
    assert(Logger); //Make sure we have global logger.
    assert(itsL>0);
    assert(isValidSpin(S));

    //InitSitesAndBonds is called from the derived class
}

int MPSImp::GetCanonicalD1(int a, int DMax)
{
    int iexp = a<=itsL/2 ? a-1 : itsL-a+1;
    return Min(static_cast<int>(pow(itsd,iexp)),DMax);
}

int MPSImp::GetCanonicalD2(int a, int DMax)
{
    int iexp = a<=itsL/2 ? a : itsL-a;
    return Min(static_cast<int>(pow(itsd,iexp)),DMax);
}

void MPSImp::InitSitesAndBonds(int D,double epsSV)
{
    //
    //  Create bond objects
    //
    itsBonds.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    for (int i=1; i<itsL; i++)
        itsBonds.push_back(new Bond(D,epsSV));
    //
    //  Create Sites
    //
    itsSites.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    itsSites.push_back(new MPSSite(PLeft,NULL,itsBonds[1],itsd,
                       GetCanonicalD1(1,D),GetCanonicalD2(1,D)));
    for (int i=2; i<=itsL-1; i++)
        itsSites.push_back(new MPSSite(PBulk,itsBonds[i-1],itsBonds[i],itsd,
                           GetCanonicalD1(i,D),GetCanonicalD2(i,D)));
    itsSites.push_back(new MPSSite(PRight,itsBonds[itsL-1],NULL,itsd,
                       GetCanonicalD1(itsL,D),GetCanonicalD2(itsL,D)));
    //
    //  Tell each bond about its left and right sites.
    //
    for (int i=1; i<itsL; i++)
        itsBonds[i]->SetSites(itsSites[i],itsSites[i+1]);
}



MPSImp::~MPSImp()
{
}

MPS* MPSImp::Clone() const
{
    return new MPSImp(*this);
}


void MPSImp::InitializeWith(State state)
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



void MPSImp::IncreaseBondDimensions(int D)
{
    int Dideal=1;
    int ia=1;
    for (; ia<=itsL/2; ia++)
    {
        int D1=Min(Dideal,D);
        int D2=Min(Dideal*itsd,D);
        itsSites[ia]->NewBondDimensions(D1,D2,true);
        assert(ia<itsL);
        itsBonds[ia]->NewBondDimension(D2);
        Dideal*=itsd;
    }
    Dideal=1;
    for (int ib=itsL; ib>=ia; ib--)
    {
        int D2=Min(Dideal,D);
        int D1=Min(Dideal*itsd,D);
        itsSites[ib]->NewBondDimensions(D1,D2,true); //Very important to save data
        assert(ib>1);
        itsBonds[ib-1]->NewBondDimension(D1);
        Dideal*=itsd;
    }

}


//--------------------------------------------------------------------------------------
//
//    Reporting
//
int MPSImp::GetMaxD() const
{
    int D=0;
    SiteLoop(ia) D=Max(D,itsSites[ia]->GetD2());
    return D;
}

void MPSImp::Report(std::ostream& os) const
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
    for (int ib=1; ib<itsL; ib++)
    {
        os << std::setw(3) << ib << "  ";
        itsBonds[ib]->Report(os);
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

}; // namespace
