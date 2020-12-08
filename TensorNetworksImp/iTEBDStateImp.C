#include "TensorNetworksImp/iTEBDStateImp.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/TNSLogger.H"

#include "TensorNetworksImp/Bond.H"

#include "NumericalMethods/ArpackEigenSolver.H"
#include "NumericalMethods/PrimeEigenSolver.H"
#include "NumericalMethods/LapackEigenSolver.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include <iomanip>

namespace TensorNetworks
{

iTEBDStateImp::iTEBDStateImp(int L,double S, int D,double normEps,double epsSV)
    : MPSImp(L,S,DLeft,normEps)
{
    InitSitesAndBonds(D,epsSV);
    ReCenter(1);
}

iTEBDStateImp::~iTEBDStateImp()
{
    itsBonds[0]=0; //Avoid double deletion in optr_vector destructor
    //dtor
}


void iTEBDStateImp::InitSitesAndBonds(int D,double epsSV)
{
    //
    //  Create bond objects
    //
    itsBonds.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    for (int i=1; i<=itsL; i++)
        itsBonds.push_back(new Bond(D,epsSV));
    itsBonds[0]=itsBonds[itsL];  //Periodic boundary conditions
    //
    //  Create Sites
    //
    itsSites.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    for (int i=1; i<=itsL-1; i++)
        itsSites.push_back(new MPSSite(PBulk,itsBonds[i-1],itsBonds[i],itsd,D,D));
    itsSites.push_back(new MPSSite(PRight,itsBonds[itsL-1],itsBonds[0],itsd,D,D));
    //
    //  Tell each bond about its left and right sites.
    //
    for (int i=1; i<=itsL; i++)
        itsBonds[i]->SetSites(itsSites[i],itsSites[GetModSite(i+1)]);
}

void iTEBDStateImp::InitializeWith(State s)
{
    MPSImp::InitializeWith(s);
}

void iTEBDStateImp::IncreaseBondDimensions(int D)
{
    for (int ia=1; ia<=itsL; ia++)
    {
        itsSites[ia]->NewBondDimensions(D,D,true);
        itsBonds[ia]->NewBondDimension(D);
    }
}


void iTEBDStateImp::ReCenter(int isite) const
{
    s1=Sites(isite,this);
    assert(s1.siteA!=s1.siteB);
    assert(s1.bondA!=s1.bondB);
    assert(s1.siteA->itsLeft_Bond==s1.bondB);
    assert(s1.siteB->itsLeft_Bond==s1.bondA);
    assert(lambdaA().size()>0);
    assert(lambdaB().size()>0);
    assert(Max(lambdaA())>0.0);
    assert(Max(lambdaB())>0.0);
}


iTEBDStateImp::Sites::Sites(int leftSite, const iTEBDStateImp* iTEBD)
    : leftSiteNumber(leftSite)
    , siteA(iTEBD->itsSites[iTEBD->GetModSite(leftSite  )])
    , siteB(iTEBD->itsSites[iTEBD->GetModSite(leftSite+1)])
    , bondA(siteA->itsRightBond)
    , bondB(siteB->itsRightBond)
    , GammaA(&siteA->itsMs)
    , GammaB(&siteB->itsMs)
    , lambdaA(&bondA->GetSVs())
    , lambdaB(&bondB->GetSVs())
{

}

iTEBDStateImp::Sites::Sites()
    : leftSiteNumber(1)
    , siteA(nullptr)
    , siteB(nullptr)
    , bondA(nullptr)
    , bondB(nullptr)
//    , GammaA()
//    , GammaB()
//    , lambdaA()
//    , lambdaB()
{

}




void iTEBDStateImp::Canonicalize(Direction lr)
{
    ForLoop(lr)
      MPSImp::CanonicalizeSite1(lr,ia,0); //Stores A1-lambda1-A2-lambda2
    ForLoop(lr)
      MPSImp::CanonicalizeSite2(lr,ia,0); //Convert to Gamma1-lambda1-Gamma2-lambda2

}

//void iTEBDStateImp::NormalizeAndCompress(Direction LR,int Dmax,double epsMin);
int iTEBDStateImp::GetModSite(int isite) const
{
    int modSite=((isite-1)%itsL)+1;
    assert(modSite>=1);
    assert(modSite<=itsL);
    return modSite;
}




//
//  Same as MPS report except we report one more bond
//
void iTEBDStateImp::Report(std::ostream& os) const
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

}
