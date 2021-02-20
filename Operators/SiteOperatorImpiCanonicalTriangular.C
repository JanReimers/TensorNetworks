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
double SiteOperatorImp::ContractT(int w11, int w12, int w21, int w22) const
{
    return Contract(w12,w11,w22,w21);
}

double SiteOperatorImp::Contract(int w11, int w12, int w21, int w22) const
{
    double r1=0.0;
    for (int m=0; m<itsd; m++)
    for (int n=0; n<itsd; n++)
        r1+=itsWOvM(w11,w12)(m,n)*itsWOvM(w21,w22)(m,n);

    return r1/itsd; //Divide by Tr[I]
}

MatrixRT SiteOperatorImp::BuildK(int M) const
{
    MatrixRT K(M,M),I(M,M);
    Unit(I);
    for (int b=0;b<=M-1;b++)
    for (int a=0;a<=M-1;a++)
    {
        K(b+1,a+1)=I(b+1,a+1)-Contract(b,a,M,M);
    }
    return K;
}
VectorRT SiteOperatorImp::Buildc(int M) const
{
    VectorRT c(M);
    Fill(c,0.0);
    for (int b=0;b<=M-1;b++)
    for (int a=0;a<=M-1;a++)
    {
        c(b+1)+=Contract(M,b,M,a);
    }
    return c;
}

void SiteOperatorImp::GaugeTransform(const MatrixRT& R, const MatrixRT& Rinv)
{
    assert(R.GetLimits()==Rinv.GetLimits());
    assert(IsUnit(R*Rinv,1e-13));
    TriType ul=itsWOvM.GetUpperLower();
    itsWOvM=MatrixOR(Transpose(R*Transpose(itsWOvM)*Rinv));
    itsWOvM.SetUpperLower(ul);
    SyncOtoW();

}

double SiteOperatorImp::Contract_sM(int M) const
{
    double dM=ContractT(M,M,M,M);
    assert(dM<1.0);
    assert(dM>=0.0);
    double sM=0.0;
    for (int a=0;a<=M;a++)
        sM+=ContractT(a,M,a,M);
    cout << "sM,dM = " << sM << " " << dM << endl;
    assert(sM>=0.0);
    sM/=(1-dM);
    assert(sM>=0.0);
    return sqrt(sM);

}
double SiteOperatorImp::Contract_sM1(int M) const
{
    int X=itsDw.Dw1-2;
    double sM=0.0;
    for (int a=1;a<=X+1;a++)
        sM+=ContractT(M,a,M,a);
    return sqrt(sM);

}
void SiteOperatorImp::iCanonicalFormTriangular(Direction lr)
{
    assert(itsDw.Dw1==itsDw.Dw2); //Make sure we are square
    int X=itsDw.Dw1-2; //Chi
    MatLimits lim(0,X+1,0,X+1);
    MatrixRT RT(lim); //Accumulated gauge transform
    LinearSolver<double>* solver=new LapackLinearSolver<double>();;
    for (int M=1;M<=X;M++)
    {
//        cout << "Init del=" << ContractDel(DLeft) << endl;
        MatrixRT K=BuildK(M);
        VectorRT c=Buildc(M);
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
        double sM=Contract_sM1(M);
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
