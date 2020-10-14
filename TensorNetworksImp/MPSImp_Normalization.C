#include "TensorNetworksImp/MPSImp.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/TNSLogger.H"

namespace TensorNetworks
{


//-------------------------------------------------------------------------------
//
//   Normalization routines
//

void MPSImp::NormalizeSite(Direction lr,int isite)
{
    CheckSiteNumber(isite);
    std::string lrs=lr==DLeft ? "Left" : "Right";
    itsLogger->LogInfo(2,isite,"SVD "+lrs+" Normalize site ");
    itsSites[isite]->SVDNormalize(lr,NULL);
    itsLogger->LogInfo(2,isite,"SVD "+lrs+" Normalize update Bond data ");
    int bond_index=isite+( lr==DLeft ? 0 :-1);
    if (bond_index<itsL && bond_index>=1)
        UpdateBondData(bond_index);
}

void MPSImp::CanonicalizeSite(Direction lr,int isite,SVCompressorC* comp)
{
    CheckSiteNumber(isite);
    itsSites[isite]->Canonicalize(lr,comp);
}

void MPSImp::NormalizeAndCompress(Direction LR,SVCompressorC* comp)
{
    ForLoop(LR)
         NormalizeAndCompressSite(LR,ia,comp);
    if (comp && comp->Donly()) itsDmax=comp->GetDmax();
}

void MPSImp::NormalizeAndCompressSite(Direction lr,int isite,SVCompressorC* comp)
{
    CheckSiteNumber(isite);
    itsSites[isite]->SVDNormalize(lr,comp);

    int bond_index=isite+( lr==DLeft ? 0 :-1);
    if (bond_index<itsL && bond_index>=1)
        UpdateBondData(bond_index);
}

//
//  Mixed canonical  A*A*A*A...A*M(isite)*B*B...B*B
//
void MPSImp::MixedCanonical(int isite)
{
    CheckSiteNumber(isite);
    if (isite>1)
    {
        for (int ia=1; ia<isite; ia++)
            NormalizeSite(DLeft,ia);
    }

    if (isite<itsL)
    {
        for (int ia=itsL; ia>isite; ia--)
            NormalizeSite(DRight,ia);
    }
}

}; // namespace
