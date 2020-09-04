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
        if (newD2<itsD2) Reshape(itsD1,newD2,true); //But also don't grow D2
        Rescale(sqrt(std::real(GetNorm(lr)(1,1))));
        return;
    }
    if(lr==TensorNetworks::DLeft && !itsRightBond)
    {
        assert(itsLeft_Bond);
        assert(itsD2==1);
        int newD1=Max(itsLeft_Bond->GetRank(),itsd); //Don't shrink below p
        if (newD1<itsD1) Reshape(newD1,itsD2,true); //But also don't grow D1
        Rescale(sqrt(std::real(GetNorm(lr)(1,1))));
        return;
    }

    //We are in the bulk
    VectorT s; // This get passed from one site to the next.
    MatrixCT A=Reshape(lr);
    int N=Min(A.GetNumRows(),A.GetNumCols());
    s.SetLimits(N);
    MatrixCT V(N,A.GetNumCols());
    CSVDecomp(A,s,V); //Solves A=U * s * Vdagger  returns V not Vdagger

    MatrixCT UV;// This get transferred through the bond to a neighbouring site.

    switch (lr)
    {
    case TensorNetworks::DRight:
    {
        UV=A;
        Reshape(lr,Transpose(conj(V)));  //A is now Vdagger
        break;
    }
    case TensorNetworks::DLeft:
    {
        UV=Transpose(conj(V)); //Set Vdagger
        Reshape(lr,A);  //A is now U
        break;
    }
    }
    GetBond(lr)->SVDTransfer(lr,s,UV);
}

void MPSSite::SVDNormalize(TensorNetworks::Direction lr, int Dmax, double epsMin)
{
    // Handle edge cases first
    if (lr==TensorNetworks::DRight && !itsLeft_Bond)
    {
        assert(itsRightBond);
        int newD2=itsRightBond->GetRank();
        Reshape(itsD1,newD2,true);
        Rescale(sqrt(std::real(GetNorm(lr)(1,1))));
        return;
    }
    if(lr==TensorNetworks::DLeft && !itsRightBond)
    {
        assert(itsLeft_Bond);
        int newD1=itsLeft_Bond->GetRank();
        Reshape(newD1,itsD2,true);
        Rescale(sqrt(std::real(GetNorm(lr)(1,1))));
        return;
    }

    //We are in the bulk
    VectorT s; // This get passed from one site to the next.
    MatrixCT A=Reshape(lr);
    int N=Min(A.GetNumRows(),A.GetNumCols());
    s.SetLimits(N);
    MatrixCT V(N,A.GetNumCols());
    CSVDecomp(A,s,V); //Solves A=U * s * Vdagger  returns V not Vdagger
    // At this point we have N singular values but we only Dmax of them or only the ones >=epsMin;
    int D=Dmax>0 ? Min(N,Dmax) : N; //Ignore Dmax if it is 0
    // Shrink so that all s(is<=D)>=epsMin;
    for (int is=D; is>=1; is--)
        if (s(is)>epsMin)
        {
            D=is;
            break;
        }
//    cout << "Smin=" << s(D) << "  Sum of rejected singular values=" << Sum(s.SubVector(D+1,s.size())) << endl;
//    cout << "Before compression Sum s=" << Sum(s) << endl;
    double Sums=Sum(s);
    assert(Sums>0.0);
    s.SetLimits(D,true);  // Resize s
    A.SetLimits(A.GetNumRows(),D,true);
    V.SetLimits(V.GetNumRows(),D,true);
    assert(Sum(s)>0.0);
    double rescaleS=Sums/Sum(s);
    s*=rescaleS;
//    cout << "After compression  Sum s=" << Sum(s) << endl;

    MatrixCT UV;// This get transferred through the bond to a neighbouring site.
    switch (lr)
    {
        case TensorNetworks::DRight:
        {
            UV=A;
            Reshape(lr,Transpose(conj(V)));  //A is now Vdagger
            break;
        }
        case TensorNetworks::DLeft:
        {
            UV=Transpose(conj(V)); //Set Vdagger
            Reshape(lr,A);  //A is now U
            break;
        }
    }
    assert(GetNormStatus(1e-12)!='M');
    GetBond(lr)->SVDTransfer(lr,s,UV);
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
        Reshape(maxAllowedD1,maxAllowedD2,true);
        reshape=true;
    }
    return reshape;
}

