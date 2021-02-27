#include "Operators/SiteOperatorImp.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/TNSLogger.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include "NumericalMethods/LapackQRSolver.H"
#include "NumericalMethods/LapackLinearSolver.H"
#include "oml/diagonalmatrix.h"

namespace TensorNetworks
{

using std::cout;
using std::endl;
//
//  <Wdagger(w11,w12),W(w21,w22)>.  The ws are zero based.
//
double SiteOperatorImp::ContractUL(int w11, int w12, int w21, int w22,MPOForm f) const
{
    if (f==RegularUpper)
        return itsWs.GetTrace(w12,w11,w21,w22);
    else if (f==RegularLower)
        return itsWs.GetTrace(w11,w12,w22,w21);
    else
        assert(false);
    return 0.0;
}

MatrixRT SiteOperatorImp::BuildK(int M,MPOForm f) const
{
    MatrixRT K(M,M),I(M,M);
    Unit(I);
    for (int b=0;b<=M-1;b++)
    for (int a=0;a<=M-1;a++)
    {
        K(b+1,a+1)=I(b+1,a+1)-ContractUL(a,b,M,M,f);
    }
    return K;
}
VectorRT SiteOperatorImp::Buildc(int M,MPOForm f) const
{
    VectorRT c(M);
    Fill(c,0.0);
    for (int b=0;b<=M-1;b++)
    for (int a=0;a<=M-1;a++)
    {
        c(b+1)+=ContractUL(b,M,a,M,f);
    }
    return c;
}

void SiteOperatorImp::GaugeTransform(const MatrixRT& R, const MatrixRT& Rinv)
{
    assert(R.GetLimits()==Rinv.GetLimits());
    assert(IsUnit(R*Rinv,1e-13));
    MPOForm f=itsWs.GetForm();
    if (f==RegularUpper)
        itsWs=R*itsWs*Rinv;
    else if (f==RegularLower)
        itsWs=Transpose(R*Transpose(itsWs)*Rinv); //ul might get lost in this step.
    else
        assert(false);
    assert(itsWs.GetForm()==f);
    SetLimits();

}

double SiteOperatorImp::Contract_sM(int M,MPOForm f) const
{
    double dM=ContractUL(M,M,M,M,f);
    assert(dM<1.0);
    assert(dM>=0.0);
    double sM=0.0;
    for (int a=0;a<=M;a++)
        sM+=ContractUL(a,M,a,M,f);
    cout << "sM,dM = " << sM << " " << dM << endl;
    assert(sM>=0.0);
    sM/=(1-dM);
    assert(sM>=0.0);
    return sqrt(sM);

}
double SiteOperatorImp::Contract_sM1(int M,MPOForm f) const
{
    int X=itsOpRange.Dw1-2;
    double sM=0.0;
    for (int a=1;a<=X+1;a++)
        sM+=ContractUL(M,a,M,a,f);
    return sqrt(sM);

}
void SiteOperatorImp::iCanonicalFormTriangular(Direction lr)
{
    assert(itsOpRange.Dw1==itsOpRange.Dw2); //Make sure we are square
    int X=itsOpRange.Dw1-2; //Chi
    MPOForm f=itsWs.GetForm();
    MatLimits lim(0,X+1,0,X+1);
    MatrixRT RT(lim); //Accumulated gauge transform
    LinearSolver<double>* solver=new LapackLinearSolver<double>();;
    for (int M=1;M<=X;M++)
    {
//        cout << "Init del=" << ContractDel(DLeft) << endl;
        MatrixRT K=BuildK(M,f);
        VectorRT c=Buildc(M,f);
//        cout << "M=" << M << endl;
//        cout << "K=" << K << endl;
//        cout << "c=" << c << endl;
        VectorRT r=solver->SolveLowerTri(K,c);
//        cout << "r=" << r << endl;
        MatrixRT R(lim),Rinv(lim);
        Unit(R);
        for (int b=0;b<=M-1;b++)
            R(b,M)=r(b+1);
//        cout << "R=" << R << endl;
        Unit(Rinv);
        for (int b=0;b<=M-1;b++)
            Rinv(b,M)=-r(b+1);
//        cout << "Rinv=" << Rinv << endl;
//        cout << "R*Rinv=" << R*Rinv << endl;

        GaugeTransform(R,Rinv);
//        cout << "After gauge del=" << ContractDel(DLeft) << endl;
        RT=R*RT;
//        {
//            MatrixRT QL=ReshapeV(DLeft);
//            cout << "Before norm QT*Q=" << Transpose(QL)*QL << endl;
//        }
        double sM=Contract_sM1(M,f);
        assert(fabs(sM)>1e-14);
//        cout << "sM^2=" << sM*sM << endl;
        if (fabs(sM)>1e-14)
        {
            Unit(R);
            Unit(Rinv);
            R   (M,M)=1.0/sM;
            Rinv(M,M)=sM;
            GaugeTransform(R,Rinv);
            RT=R*RT;
        }
        else
        {
            assert(false);
            //Need code to remove rows and columns
        }
//        cout << "After norm del=" << ContractDel(DLeft) << endl;
//        {
//            MatrixRT QL=ReshapeV(DLeft);
//            cout << "After norm QT*Q=" << Transpose(QL)*QL << endl;
//        }

    }
    delete solver;
}


} //namespace
