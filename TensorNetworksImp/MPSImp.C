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
    , itsLogger      (mps.itsLogger)
    , itsSitesMesh   (0)
    , itsBondsMesh   (0)
{
    assert(itsDmax>0);
    InitSitesAndBonds();
    InitPlotting();
    for (int ia=1; ia<=itsL; ia++)
        itsSites[ia]->CloneState(mps.itsSites[ia]); //Transfer wave function data
    for (int ia=1; ia<itsL; ia++)
        itsBonds[ia]->CloneState(mps.itsBonds[ia]); //Transfer wave function data
}

MPSImp::MPSImp(int L, double S, int D,TensorNetworks::Direction lr,double normEps,TNSLogger* s)
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
    assert(this->itsDmax>0);
    return new MPSImp(*this);
}





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

    itsDmax=D;
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

c_str MPSImp::SiteMessage(const std::string& message,int isite)
{
    static std::string ret;
    ret=message+std::to_string(isite);
    return ret.c_str();
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

