#include "TensorNetworksImp/MPSSite.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/Dw12.H"
#include "NumericalMethods/LapackSVDSolver.H"
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
        if (Dmax>0 && Dmax<D) D=Dmax;
    }
    auto [U,s,Vdagger]=solver.Solve(A,1e-13,D); //Solves A=U * s * Vdagger
    double compessionError=0.0;
    if (comp) compessionError=comp->Compress(U,s,Vdagger);

    switch (lr)
    {
    case DRight:
    {
        GetBond(lr)->SVDTransfer(lr,compessionError,s,U);
        ReshapeAfter_SVD(lr,Vdagger);  //A is now Vdagger
        itsNormStatus=NormStatus::B;
        break;
    }
    case DLeft:
    {
        GetBond(lr)->SVDTransfer(lr,compessionError,s,Vdagger);
        ReshapeAfter_SVD(lr,U);  //A is now U
        itsNormStatus=NormStatus::A;
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


void MPSSite::Canonicalize1(Direction lr,SVCompressorC* comp)
{
    MatrixCT A=ReshapeBeforeSVD(lr);
    auto [U,s,Vdagger]=oml_CSVDecomp(A); //Solves A=U * s * Vdagger  returns V not Vdagger
    double compressionError=0.0;
    if (comp) compressionError=comp->Compress(U,s,Vdagger);
    double s2=s.GetDiagonal()*s.GetDiagonal();
    s/=sqrt(s2);

    switch (lr)
    {
    case DRight:
    {
        GetBond(lr)->CanonicalTransfer(lr,compressionError,s,U);
        ReshapeAfter_SVD(lr,Vdagger);  //A is now Vdagger
        itsNormStatus=NormStatus::B;
        break;
    }
    case DLeft:
    {
        GetBond(lr)->CanonicalTransfer(lr,compressionError,s,Vdagger);
        ReshapeAfter_SVD(lr,U);  //A is now U
        itsNormStatus=NormStatus::A;
        break;
    }
    }
}

void MPSSite::Canonicalize2(Direction lr,SVCompressorC* comp)
{
    Bond* bond=GetBond(Invert(lr));
    assert(bond);
    DiagonalMatrixRT lambda=bond->GetSVs();
    assert(lambda.size()==itsD1);
    assert(Min(lambda)>1e-10);
    DiagonalMatrixRT linv=1.0/lambda;

    switch (lr)
    {
        case DRight:
        {
            for (int n=0; n<itsd; n++)
                itsMs[n]=MatrixCT(itsMs[n]*linv);
//            for (auto m:itsMs) m=MatrixCT(m*linv);
            itsNormStatus=NormStatus::GammaRight;
            break;
        }
        case DLeft:
        {
            for (int n=0; n<itsd; n++)
                itsMs[n]=MatrixCT(linv*itsMs[n]);
//            for (auto m:itsMs)
//            {
//                cout << m << endl;
//                m=MatrixCT(linv*m);
//                cout << m << endl;
//            }
            itsNormStatus=NormStatus::GammaLeft;
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
