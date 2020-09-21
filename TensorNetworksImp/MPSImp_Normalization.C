#include "TensorNetworksImp/MPSImp.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/TNSLogger.H"
#include "Containers/Matrix4.H"
#include "Functions/Mesh/PlotableMesh.H"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

int MPSImp::GetD1(int a, int L, int d, int DMax)
{
    int D=DMax;
    if (a<=L/2)
        D=Min(static_cast<int>(pow(d,a-1)),DMax); //LHS
    else
        D=Min(static_cast<int>(pow(d,L-a+1)),DMax);  //RHS
    return D;
}

int MPSImp::GetD2(int a, int L, int d, int DMax)
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
//   Normalization routines
//
void MPSImp::Normalize(TensorNetworks::Direction LR)
{
    ForLoop(LR)
        NormalizeSite(LR,ia);
}


void MPSImp::NormalizeSite(TensorNetworks::Direction lr,int isite)
{
//    NormalizeAndCompressSite(lr,isite,NULL);
    CheckSiteNumber(isite);
    std::string lrs=lr==TensorNetworks::DLeft ? "Left" : "Right";
    itsLogger->LogInfo(2,isite,"SVD "+lrs+" Normalize site ");
    itsSites[isite]->SVDNormalize(lr);
    itsLogger->LogInfo(2,isite,"SVD "+lrs+" Normalize update Bond data ");
    int bond_index=isite+( lr==TensorNetworks::DLeft ? 0 :-1);
    if (bond_index<itsL && bond_index>=1)
        UpdateBondData(bond_index);
}

void MPSImp::CanonicalizeSite(TensorNetworks::Direction lr,int isite)
{
    CheckSiteNumber(isite);
    itsSites[isite]->Canonicalize(lr);
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

void MPSImp::NormalizeAndCompress(TensorNetworks::Direction LR,SVCompressor* comp)
{
//    SetCanonicalBondDimensions(Invert(LR)); //Sweep backwards and set proper bond dimensions
    ForLoop(LR)
        NormalizeAndCompressSite(LR,ia,comp);
    if (comp) itsDmax=comp->GetDmax();
}

void MPSImp::NormalizeAndCompressSite(TensorNetworks::Direction lr,int isite,SVCompressor* comp)
{
    CheckSiteNumber(isite);
//    cout << "----- Normalize and Compress site " << isite << " " << GetNormStatus() << " -----" << endl;
    std::string lrs=lr==TensorNetworks::DLeft ? "Left" : "Right";
    itsLogger->LogInfo(2,isite,"SVD "+lrs+" Normalize site ");
    //cout << "SVD " << lrs << " site " << isite << endl;
    itsSites[isite]->SVDNormalize(lr,comp);

    itsLogger->LogInfo(2,isite,"SVD "+lrs+" Normalize update Bond data ");
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
