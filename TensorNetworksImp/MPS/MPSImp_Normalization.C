#include "TensorNetworksImp/MPS/MPSImp.H"
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
    itsSites[isite]->SVDNormalize(lr,NULL);
    int bond_index=isite+( lr==DLeft ? 0 :-1);
    if (bond_index<itsL && bond_index>=1)
        UpdateBondData(bond_index);
}

void MPSImp::CanonicalizeSite1(Direction lr,int isite,SVCompressorC* comp)
{
    CheckSiteNumber(isite);
    itsSites[isite]->Canonicalize1(lr,comp);
}
void MPSImp::CanonicalizeSite2(Direction lr,int isite,SVCompressorC* comp)
{
    CheckSiteNumber(isite);
    itsSites[isite]->Canonicalize2(lr,comp);
}

void MPSImp::NormalizeAndCompress(Direction LR,SVCompressorC* comp)
{
    ForLoop(LR)
         NormalizeAndCompressSite(LR,ia,comp);
}
void MPSImp::NormalizeQR(Direction LR)
{
    ForLoop(LR)
         NormalizeQRSite(LR,ia);
}

void MPSImp::NormalizeAndCompressSite(Direction lr,int isite,SVCompressorC* comp)
{
    CheckSiteNumber(isite);
    itsSites[isite]->SVDNormalize(lr,comp);

    int bond_index=isite+( lr==DLeft ? 0 :-1);
    if (bond_index<itsL && bond_index>=1)
        UpdateBondData(bond_index);
}

void MPSImp::NormalizeQRSite(Direction lr,int isite)
{
    CheckSiteNumber(isite);
    itsSites[isite]->NormalizeQR(lr);

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
