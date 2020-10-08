#include "TensorNetworksImp/MPSSite.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/Dw12.H"
#include "oml/minmax.h"
#include "oml/cnumeric.h"
//#include "oml/vector_io.h"
#include "oml/random.h"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

void MPSSite::SVDNormalize(TensorNetworks::Direction lr)
{
    SVDNormalize(lr,NULL);
}

void MPSSite::SVDNormalize(TensorNetworks::Direction lr, SVCompressorC* comp)
{
    // Handle edge cases first
    if (lr==TensorNetworks::DRight && !itsLeft_Bond)
    {
        assert(itsRightBond);
        assert(itsD1==1);
        int newD2=Max(itsRightBond->GetRank(),itsd); //Don't shrink below p
        if (newD2<itsD2) NewBondDimensions(itsD1,newD2,true); //But also don't grow D2
        Rescale(sqrt(std::real(GetNorm(lr)(1,1))));
        return;
    }
    if(lr==TensorNetworks::DLeft && !itsRightBond)
    {
        assert(itsLeft_Bond);
        assert(itsD2==1);
        int newD1=Max(itsLeft_Bond->GetRank(),itsd); //Don't shrink below p
        if (newD1<itsD1) NewBondDimensions(newD1,itsD2,true); //But also don't grow D1
        Rescale(sqrt(std::real(GetNorm(lr)(1,1))));
        return;
    }

    auto [U,s,Vdagger]=oml_CSVDecomp(ReshapeBeforeSVD(lr)); //Solves A=U * s * Vdagger  returns V not Vdagger
    if (comp) comp->Compress(U,s,Vdagger);
//    cout << "Limits for U,s,Vdagger=" << U.GetLimits() << " " << s.GetLimits() << " " << Vdagger.GetLimits() << endl;
    switch (lr)
    {
        case TensorNetworks::DRight:
        {
            GetBond(lr)->SVDTransfer(lr,s,U);
            ReshapeAfter_SVD(lr,Vdagger);  //A is now Vdagger
            break;
        }
        case TensorNetworks::DLeft:
        {
            GetBond(lr)->SVDTransfer(lr,s,Vdagger);
            ReshapeAfter_SVD(lr,U);  //A is now U
            break;
        }
    }
    assert(GetNormStatus(1e-12)!='M');
}

void MPSSite::Rescale(double norm)
{
    for (int n=0; n<itsd; n++) itsMs[n]/=norm;
}

bool MPSSite::SetCanonicalBondDimensions(int maxAllowedD1,int maxAllowedD2)
{
    bool reshape=false;
    if (itsD1>maxAllowedD1 || itsD2 >maxAllowedD2)
    {
        assert(itsD1>=maxAllowedD1);
        assert(itsD2>=maxAllowedD2);
        NewBondDimensions(maxAllowedD1,maxAllowedD2,true);
        reshape=true;
    }
    return reshape;
}

void MPSSite::Canonicalize(TensorNetworks::Direction lr)
{
    MatrixCT A=ReshapeBeforeSVD(lr);
//    int N=Min(A.GetNumRows(),A.GetNumCols());
//    VectorRT s(N); // This get passed from one site to the next.
//    MatrixCT V(N,A.GetNumCols());
    auto [U,s,Vdagger]=oml_CSVDecomp(A); //Solves A=U * s * Vdagger  returns V not Vdagger

    switch (lr)
    {
        case TensorNetworks::DRight:
        {
            GetBond(lr)->CanonicalTransfer(lr,s,U);
            ReshapeAfter_SVD(lr,Vdagger);  //A is now Vdagger
            break;
        }
        case TensorNetworks::DLeft:
        {
            GetBond(lr)->CanonicalTransfer(lr,s,Vdagger);
            ReshapeAfter_SVD(lr,U);  //A is now U
            break;
        }
    }
}


