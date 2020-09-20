#include "TensorNetworksImp/MPSSite.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/SiteOperator.H"
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

    auto [U,s,V]=CSVDecomp(ReshapeBeforeSVD(lr)); //Solves A=U * s * Vdagger  returns V not Vdagger
    MatrixCT Vdagger=Transpose(conj(V));

    switch (lr)
    {
        case TensorNetworks::DRight:
        {
            GetBond(lr)->SVDTransfer(lr,s,U);
            ReshapeAfter_SVD(lr,Vdagger);
            break;
        }
        case TensorNetworks::DLeft:
        {
            GetBond(lr)->SVDTransfer(lr,s,Vdagger);
            ReshapeAfter_SVD(lr,U);
            break;
        }
    }
}

void MPSSite::SVDNormalize(TensorNetworks::Direction lr, int Dmax, double epsMin)
{
    // Handle edge cases first
    if (lr==TensorNetworks::DRight && !itsLeft_Bond)
    {
        assert(itsRightBond);
        int newD2=itsRightBond->GetRank();
        NewBondDimensions(itsD1,newD2,true);
        Rescale(sqrt(std::real(GetNorm(lr)(1,1))));
        return;
    }
    if(lr==TensorNetworks::DLeft && !itsRightBond)
    {
        assert(itsLeft_Bond);
        int newD1=itsLeft_Bond->GetRank();
        NewBondDimensions(newD1,itsD2,true);
        Rescale(sqrt(std::real(GetNorm(lr)(1,1))));
        return;
    }

    auto [U,s,V]=CSVDecomp(ReshapeBeforeSVD(lr)); //Solves A=U * s * Vdagger  returns V not Vdagger
    int N=s.GetNumRows();

    // At this point we have N singular values but we only Dmax of them or only the ones >=epsMin;
    int D=Dmax>0 ? Min(N,Dmax) : N; //Ignore Dmax if it is 0
    // Shrink so that all s(is<=D)>=epsMin;
    for (int is=D; is>=1; is--)
        if (s(is,is)>epsMin)
        {
            D=is;
            break;
        }
//    cout << "Smin=" << s(D) << "  Sum of rejected singular values=" << Sum(s.SubVector(D+1,s.size())) << endl;
//    cout << "Before compression Sum s=" << Sum(s) << endl;
    double Sums=Sum(s.GetDiagonal());
    assert(Sums>0.0);
    s.SetLimits(D,true);  // Resize s
    U.SetLimits(U.GetNumRows(),D,true);
    V.SetLimits(V.GetNumRows(),D,true);
    assert(Sum(s.GetDiagonal())>0.0);
    double rescaleS=Sums/Sum(s.GetDiagonal());
    s*=rescaleS;
//    cout << "After compression  Sum s=" << Sum(s) << endl;

    MatrixCT Vdagger=Transpose(conj(V));
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
    int N=Min(A.GetNumRows(),A.GetNumCols());
    VectorT s(N); // This get passed from one site to the next.
    MatrixCT V(N,A.GetNumCols());
    CSVDecomp(A,s,V); //Solves A=U * s * Vdagger  returns V not Vdagger

    MatrixCT UV;// This get transferred through the bond to a neighbouring site.

    switch (lr)
    {
    case TensorNetworks::DRight:
    {
        UV=A;
        ReshapeAfter_SVD(lr,Transpose(conj(V)));  //A is now Vdagger
        break;
    }
    case TensorNetworks::DLeft:
    {
        UV=Transpose(conj(V)); //Set Vdagger
        ReshapeAfter_SVD(lr,A);  //A is now U
        break;
    }
    }
    GetBond(lr)->CanonicalTransfer(lr,DiagonalMatrix(s),UV);
}


