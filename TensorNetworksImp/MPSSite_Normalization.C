#include "TensorNetworksImp/MPSSite.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/Dw12.H"
#include "NumericalMethods/LapackSVD.H"
#include "NumericalMethods/ArpackEigenSolver.H"
#include "Containers/Matrix4.H"
#include "oml/cnumeric.h"
#include "oml/random.h"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

namespace TensorNetworks
{

void MPSSite::SVDNormalize(Direction lr, SVCompressorC* comp)
{
    // Handle edge cases first
    if (lr==DRight && !itsLeft_Bond)
    {
        assert(itsRightBond);
        assert(itsD1==1);
        int newD2=Max(itsRightBond->GetD(),itsd); //Don't shrink below p
        if (newD2<itsD2) NewBondDimensions(itsD1,newD2,true); //But also don't grow D2
        Rescale(sqrt(std::real(GetNorm(lr)(1,1))));
        return;
    }
    if(lr==DLeft && !itsRightBond)
    {
        assert(itsLeft_Bond);
        assert(itsD2==1);
        int newD1=Max(itsLeft_Bond->GetD(),itsd); //Don't shrink below p
        if (newD1<itsD1) NewBondDimensions(newD1,itsD2,true); //But also don't grow D1
        Rescale(sqrt(std::real(GetNorm(lr)(1,1))));
        return;
    }
    MatrixCT A=ReshapeBeforeSVD(lr);
    LapackSVDSolver<dcmplx> solver;
    int D=Min(A.GetNumRows(),A.GetNumCols());
    if (comp)
    {
        int Dmax=comp->GetDmax();
//        cout << "D Dmax=" << D << " " << Dmax << endl;
        if (Dmax>0 && Dmax<D) D=Dmax;
    }
    auto [U,s,Vdagger]=solver.Solve(A,D,1e-10); //Solves A=U * s * Vdagger
//    auto [U,s,Vdagger]=oml_CSVDecomp(ReshapeBeforeSVD(lr)); //Solves A=U * s * Vdagger
    double integratedS2=0.0;
    if (comp) integratedS2=comp->Compress(U,s,Vdagger);

    switch (lr)
    {
        case DRight:
        {
            GetBond(lr)->SVDTransfer(lr,integratedS2,s,U);
            ReshapeAfter_SVD(lr,Vdagger);  //A is now Vdagger
            break;
        }
        case DLeft:
        {
            GetBond(lr)->SVDTransfer(lr,integratedS2,s,Vdagger);
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
double   MPSSite::FrobeniusNorm() const
{
    double fnorm=0;
    for (int n=0; n<itsd; n++)
    {
        double f=::FrobeniusNorm(itsMs[n]);
        fnorm+=f*f;
    }
    return sqrt(fnorm);
}


void MPSSite::Canonicalize(Direction lr,SVCompressorC* comp)
{
    MatrixCT A=ReshapeBeforeSVD(lr);
    auto [U,s,Vdagger]=oml_CSVDecomp(A); //Solves A=U * s * Vdagger  returns V not Vdagger
    double integratedS2=0.0;
    if (comp) integratedS2=comp->Compress(U,s,Vdagger);

    switch (lr)
    {
        case DRight:
        {
            GetBond(lr)->CanonicalTransfer(lr,integratedS2,s,U);
            ReshapeAfter_SVD(lr,Vdagger);  //A is now Vdagger
            break;
        }
        case DLeft:
        {
            GetBond(lr)->CanonicalTransfer(lr,integratedS2,s,Vdagger);
            ReshapeAfter_SVD(lr,U);  //A is now U
            break;
        }
    }
}

void MPSSite::iNormalize(Direction lr)
{
    MatrixCT E=GetTransferMatrix(lr).Flatten();
    EigenSolver<dcmplx>* solver=new ArpackEigenSolver<dcmplx>;
    dcmplx eigenValue(0);
    if (lr==DLeft)
    {
        auto [U,d]=solver->SolveLeft_NonSym(E,1e-13,1);
        cout << std::fixed << std::setprecision(4) << "Left  Arpack d=" << d << endl;
        eigenValue=d(1);
    }
    if (lr==DRight)
    {
        auto [U,d]=solver->SolveRightNonSym(E,1e-13,1);
        cout << std::fixed << std::setprecision(4) << "Right Arpack d=" << d << endl;
        eigenValue=d(1);
    }
    delete solver;
    assert(fabs(imag(eigenValue))<1e-10);
    Rescale(sqrt(real(eigenValue)));
}

} //namespace
