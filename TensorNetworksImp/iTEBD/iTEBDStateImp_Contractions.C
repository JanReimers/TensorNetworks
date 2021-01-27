#include "TensorNetworksImp/iTEBD/iTEBDStateImp.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/TNSLogger.H"

#include "TensorNetworksImp/MPS/Bond.H"

#include "NumericalMethods/ArpackEigenSolver.H"
#include "NumericalMethods/PrimeEigenSolver.H"
#include "NumericalMethods/LapackEigenSolver.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include "oml/cnumeric.h"
#include <iomanip>

namespace TensorNetworks
{

MPSSite::dVectorT  iTEBDStateImp::ContractTheta(const Matrix4RT& expH, ThetaType tt) const
{
    //
    //  Make sure Ds line up
    //
    assert(s1.siteA->GetD2()==s1.siteB->GetD1());
    assert(s1.siteA->GetD1()==s1.siteB->GetD2());
    int D1=s1.siteA->GetD1();
//    int D2=s1.siteA->GetD2();
    //
    //  Create empty theta tensor
    //
    dVectorT  Thetap(itsd*itsd);
    for (int n=0;n<itsd*itsd;n++)
    {
        Thetap[n].SetLimits(D1,D1);
        Fill(Thetap[n],dcmplx(0.0));
    }
    //
    //  Contract
    //
    for (int mb=0; mb<itsd; mb++)
    for (int ma=0; ma<itsd; ma++)
    {
        MatrixCT theta13  =ContractTheta(ma,mb,tt);  //Figure 14 v
        int nab=0;
        for (int nb=0; nb<itsd; nb++)
        for (int na=0; na<itsd; na++,nab++)
            if (expH(ma,na,mb,nb)!=0.0)
                Thetap[nab]+= theta13*expH(ma,na,mb,nb);
    }

    return Thetap;
}

MatrixCT iTEBDStateImp::ContractTheta(int ma, int mb,ThetaType tt) const
{
    MatrixCT theta;
    switch (tt)
    {
    case AlB:
        theta=          GammaA()[ma]*lambdaA()*GammaB()[mb];
        break;
    case lAlB:
        theta=lambdaB()*GammaA()[ma]*lambdaA()*GammaB()[mb];
        break;
    case lBlA:
        theta=lambdaA()*GammaB()[mb]*lambdaB()*GammaA()[ma];
        break;
    case lAlBl:
        theta=lambdaB()*GammaA()[ma]*lambdaA()*GammaB()[mb]*lambdaB();
        break;
    case lBlAl:
        theta=lambdaA()*GammaB()[mb]*lambdaB()*GammaA()[ma]*lambdaA();
        break;
    case rBlAr:
        theta=sqrt(lambdaA())*GammaB()[mb]*lambdaB()*GammaA()[ma]*sqrt(lambdaA());
        break;
    }
    return theta;
}

MPSSite::dVectorT  iTEBDStateImp::ContractTheta(const iMPO* o,ThetaType tt) const
{
    //
    //  Make sure everything is square
    //
    assert(s1.siteA->GetD2()==s1.siteB->GetD1());
    assert(s1.siteA->GetD1()==s1.siteB->GetD2());
//    assert(s1.siteA->GetD1()==s1.siteA->GetD2());
    int D=s1.siteA->GetD1();
    //
    //  Set up two site operators
    //
    assert(o->GetL()==2);
    const SiteOperator* soA=0;
    const SiteOperator* soB=0;
    switch (tt)
    {
    case AlB:
    case lAlB:
    case lAlBl:
        if (s1.leftSiteNumber==1)
        {
            soA=o->GetSiteOperator(1);
            soB=o->GetSiteOperator(2);
        }
        else if(s1.leftSiteNumber==2)
        {
            soA=o->GetSiteOperator(1);
            soB=o->GetSiteOperator(2);
        }
        else
            assert(false);
        break;
    case lBlA:
    case lBlAl:
    case rBlAr:
        if (s1.leftSiteNumber==1)
        {
            soA=o->GetSiteOperator(2);
            soB=o->GetSiteOperator(1);
        }
        else if(s1.leftSiteNumber==2)
        {
            soA=o->GetSiteOperator(1);
            soB=o->GetSiteOperator(2);
        }
        else
            assert(false);
        break;
    }
    assert(soA->GetDw12().Dw2==soB->GetDw12().Dw1);
    assert(soA->GetDw12().Dw1==soB->GetDw12().Dw2);
    int Dw2=soA->GetDw12().Dw2;
    int Dw1=soA->GetDw12().Dw1; //Same as Dw3
    //
    //  Create empty theta tensor
    //
    int DDw=D*Dw1;
    dVectorT  Thetap(itsd*itsd);
    for (int n=0;n<itsd*itsd;n++)
    {
        Thetap[n].SetLimits(DDw,DDw);
        Fill(Thetap[n],dcmplx(0.0));
    }
    //
    //  Contract i1 -- GammaA -- LambdaA -- GammaB -- i3
    //                    |na                  |nb
    //                    |                    |
    //         w1 --------o--------w2----------o----- w3
    //                    |                    |
    //                    |                    |
    //
    //  We really need 3 versions:
    //      1) LambdaB -- GammaA -- LambdaA -- GammaB --
    //      2)         -- GammaA -- LambdaA -- GammaB -- LambdaB
    //      3) LambdaB -- GammaA -- LambdaA -- GammaB -- LambdaB
    //
    //  When Dw===w1==w3==1 (i.e. open legs at the edges of the operator) then this is
    //  easy.  But when Dw > 1 then this become non-trivial because the indices get combined (i1,w1) (i3,w3).
    //
    for (int mb=0; mb<itsd; mb++)
    for (int ma=0; ma<itsd; ma++)
    {
//        MatrixCT theta13  =GammaA()[ma]*lambdaA()*GammaB()[mb];  //Figure 14 v
        MatrixCT theta13  =ContractTheta(ma,mb,tt);  //Figure 14 v
        int nab=0;
        for (int nb=0; nb<itsd; nb++)
        for (int na=0; na<itsd; na++,nab++)
        {
            const MatrixRT& WAmn=soA->GetW(ma,na);
            const MatrixRT& WBmn=soB->GetW(mb,nb);
            assert(WAmn.GetNumCols()==Dw2);
            assert(WBmn.GetNumRows()==Dw2);
            Matrix4CT thetaw(D,Dw1,D,Dw1);
            for (int w3=1;w3<=Dw1;w3++)
            for (int w1=1;w1<=Dw1;w1++)
            {
                double Omn(0);
                for (int w2=1; w2<=Dw2; w2++)
                    Omn+=WAmn(w1,w2)*WBmn(w2,w3);

                for (int i1=1;i1<=D;i1++)
                for (int i3=1;i3<=D;i3++)
                    thetaw(i1,w1,i3,w3)=theta13(i1,i3)*Omn;
            }
            Thetap[nab]+= thetaw.Flatten();;
       }
    }

    return Thetap;
}

MPSSite::dVectorT  iTEBDStateImp::ContractThetaDw(const iMPO* o) const
{
    //
    //  Make sure everything is square
    //
    assert(s1.siteA->GetD2()==s1.siteB->GetD1());
    assert(s1.siteA->GetD1()==s1.siteB->GetD2());
//    assert(s1.siteA->GetD1()==s1.siteA->GetD2());
    int D=s1.siteA->GetD1();
    //
    //  Set up two site operators
    //
    assert(o->GetL()==2);
    const SiteOperator* soA=o->GetSiteOperator(1);
    const SiteOperator* soB=o->GetSiteOperator(2);
    assert(soA->GetDw12().Dw2==soB->GetDw12().Dw1);
    assert(soA->GetDw12().Dw1==soB->GetDw12().Dw2);
    int Dw2=soA->GetDw12().Dw2;
    int Dw1=soA->GetDw12().Dw1; //Same as Dw3
    //
    //  Create empty theta tensor
    //
    int DDw=D*Dw1;
    dVectorT  Thetap(itsd*itsd);
    for (int n=0;n<itsd*itsd;n++)
    {
        Thetap[n].SetLimits(DDw,DDw);
        Fill(Thetap[n],dcmplx(0.0));
    }
    //
    //  Contract
    //            i1 -- GammaA -- LambdaA -- GammaB -- i3
    //           /         |na                  |nb      \_LambdaB
    //    LambdaB\         |                    |        /
    //            w1-------o--------w2----------o----- w3
    //                     |                    |
    //                     |                    |
    //
    //  When Dw===w1==w3==1 (i.e. open legs at the edges of the operator) then this is
    //  easy.  But when Dw > 1 then this become non-trivial because the indices get combined (i1,w1) (i3,w3).
    //
    for (int mb=0; mb<itsd; mb++)
    for (int ma=0; ma<itsd; ma++)
    {
        MatrixCT theta13  =ContractTheta(ma,mb,AlB);  //Figure 14 v
        int nab=0;
        for (int nb=0; nb<itsd; nb++)
        for (int na=0; na<itsd; na++,nab++)
        {
            const MatrixRT& WAmn=soA->GetW(ma,na);
            const MatrixRT& WBmn=soB->GetW(mb,nb);
            assert(WAmn.GetNumCols()==Dw2);
            assert(WBmn.GetNumRows()==Dw2);
            Matrix4CT thetaw(D,Dw1,D,Dw1);
            for (int w3=1;w3<=Dw1;w3++)
            for (int w1=1;w1<=Dw1;w1++)
            {
                double Omn(0);
                for (int w2=1; w2<=Dw2; w2++)
                    Omn+=WAmn(w1,w2)*WBmn(w2,w3);

                for (int i1=1;i1<=D;i1++)
                for (int i3=1;i3<=D;i3++)
                {
                    int iw1=D*w1+i1;
                    int iw3=D*w3+i3;
                    thetaw(i1,w1,i3,w3)=lambdaB()(iw1,iw1)*theta13(i1,i3)*Omn*lambdaB()(iw3,iw3);
                }
            }
            Thetap[nab]+= thetaw.Flatten();;
       }
    }

    return Thetap;
}

Matrix4CT iTEBDStateImp::GetTransferMatrix(const dVectorT& M)
{
    int d=M.size();
    assert(d>0);
    int D=M[0].GetNumRows();
    assert(D==M[0].GetNumCols());
    Matrix4CT E(D,D,D,D);

    for (int i1=1; i1<=D; i1++)
        for (int j1=1; j1<=D; j1++)
            for (int i3=1; i3<=D; i3++)
                for (int j3=1; j3<=D; j3++)
                {
                    dcmplx e(0);
                    for (int n=0; n<d; n++)
                        e+=M[n](i1,i3)*conj(M[n](j1,j3));
                    E(i1,j1,i3,j3)=e;
                }
    return E;
}

MatrixCT  iTEBDStateImp::GetNormMatrix(Direction lr,const dVectorT& M)  //Er*I or I*El
{
    int d=M.size();
    assert(d>0);
    int D=M[0].GetNumRows();
    assert(D==M[0].GetNumCols());
    MatrixCT N(D,D);
    switch (lr)
    {
        case DLeft:
        {
            for (int i3=1; i3<=D; i3++)
            for (int j3=1; j3<=D; j3++)
            {
                dcmplx e(0);
                for (int i1=1; i1<=D; i1++)
                    for (int n=0; n<d; n++)
                        e+=M[n](i1,i3)*conj(M[n](i1,j3));
                N(i3,j3)=e;
            }
            break;
        }
        case DRight:
        {
            for (int i1=1; i1<=D; i1++)
            for (int j1=1; j1<=D; j1++)
            {
                dcmplx e(0);
                for (int i3=1; i3<=D; i3++)
                    for (int n=0; n<d; n++)
                        e+=M[n](i1,i3)*conj(M[n](j1,i3));
                N(i1,j1)=e;
            }
            break;
        }
    }
    return N;
}

//
//  Contract GammaA()[na]*lambdaA()*GammaB()[nb]
//
iTEBDStateImp::dVectorT iTEBDStateImp::ContractAlB() const
{
    dVectorT gamma(itsd*itsd);
    int nab=0;
    for (int nb=0; nb<itsd; nb++)
        for (int na=0; na<itsd; na++,nab++)
            gamma[nab]=GammaA()[na]*lambdaA()*GammaB()[nb];
    return gamma;
}
//
//  Assume two site for now
//
Matrix4CT iTEBDStateImp::GetTransferMatrix(Direction lr) const
{
    dVectorT gamma;
    switch (lr)
    {
    case DRight :
        gamma=ContractAlB()*lambdaB();
        break;
    case DLeft :
        gamma=lambdaB()*ContractAlB();
        break;
    }
    return GetTransferMatrix(gamma);;
}


double iTEBDStateImp::GetExpectationmmnn (const Matrix4RT& Hlocal) const
{
    int oldCenter=s1.leftSiteNumber;
    double e1=GetExpectationmmnn(Hlocal,1);
    double e2=GetExpectationmmnn(Hlocal,2);
    ReCenter(oldCenter);
    return 0.5*(e1+e2);
}

double iTEBDStateImp::GetExpectationmmnn (const Matrix4RT& Hlocal, int center) const
{
    ReCenter(center);
    int D=s1.siteA->GetD1();
    assert(D==s1.siteA->GetD2());
    assert(D==s1.siteB->GetD1());
    assert(D==s1.siteB->GetD2());

    dcmplx expectation1(0.0);
    for (int na=0; na<itsd; na++)
        for (int nb=0; nb<itsd; nb++)
        {
            MatrixCT theta13_n=lambdaB()*GammaA()[na]*lambdaA()*GammaB()[nb]*lambdaB();
            assert(theta13_n.GetNumRows()==D);
            assert(theta13_n.GetNumCols()==D);
            for (int ma=0; ma<itsd; ma++)
                for (int mb=0; mb<itsd; mb++)
                {
                    MatrixCT theta13_m=conj(lambdaB()*GammaA()[ma])*lambdaA()*conj(GammaB()[mb])*lambdaB();
                    assert(theta13_m.GetNumRows()==D);
                    assert(theta13_m.GetNumCols()==D);
                    for (int i1=1; i1<=D; i1++)
                        for (int i3=1; i3<=D; i3++)
                            expectation1+=theta13_m(i1,i3)*Hlocal(ma,mb,na,nb)*theta13_n(i1,i3);
                }
        }
    if (fabs(imag(expectation1))>=1e-10)
        Logger->LogWarnV(0,"iTEBDStateImp::GetExpectation expectation=(%.5f,%.1e) has large imaginary component",real(expectation1),imag(expectation1));
    return real(expectation1);

}
double iTEBDStateImp::GetExpectationmnmn (const Matrix4RT& expH) const
{
//    assert(TestOrthogonal(1e-11));
    int D=s1.siteA->GetD1();
    assert(D==s1.siteA->GetD2());
    assert(D==s1.siteB->GetD1());
    assert(D==s1.siteB->GetD2());

    dcmplx expectation1(0.0);
    for (int na=0; na<itsd; na++)
        for (int nb=0; nb<itsd; nb++)
        {
            MatrixCT theta13_n=lambdaB()*GammaA()[na]*lambdaA()*GammaB()[nb]*lambdaB();
            assert(theta13_n.GetNumRows()==D);
            assert(theta13_n.GetNumCols()==D);
            for (int ma=0; ma<itsd; ma++)
                for (int mb=0; mb<itsd; mb++)
                {
                    MatrixCT theta13_m=conj(lambdaB()*GammaA()[ma])*lambdaA()*conj(GammaB()[mb])*lambdaB();
                    assert(theta13_m.GetNumRows()==D);
                    assert(theta13_m.GetNumCols()==D);
                    for (int i1=1; i1<=D; i1++)
                        for (int i3=1; i3<=D; i3++)
                            expectation1+=theta13_m(i1,i3)*expH(ma,na,mb,nb)*theta13_n(i1,i3);
                }
        }
    if (fabs(imag(expectation1))>=1e-10)
        Logger->LogWarnV(0,"iTEBDStateImp::GetExpectation expectation=(%.5f,%.1e) has large imaginary component",real(expectation1),imag(expectation1));
    return real(expectation1);

}

double iTEBDStateImp::GetExpectation (const iMPO* o) const
{
    int L=GetL();
    assert(L==2);
    assert(o->GetL()==L);
//
//  The first recursion relation for E^Dw is the just orthonormality condition
//
    double E1;
    ReCenter(1);
    {
//        cout << "Oerr=" << GetOrthonormalityErrors() << endl;
        assert(GetOrthonormalityErrors()<1e-11);

        dVectorT A=lambdaB()*ContractAlB();
        assert(A.size()==itsd*itsd);
        iMPO* cellMPO=o->MakeUnitcelliMPO(L);

        E1=GetExpectation(A,lambdaB(),cellMPO);
        delete cellMPO;
    }
//    double E2;
//    ReCenter(2);
//    {
//        cout << "oerr=" << GetOrthonormalityErrors() << endl;
//        assert(GetOrthonormalityErrors()<1e-11);
//
//        dVectorT A=lambdaB()*ContractAlB();
//        assert(A.size()==itsd*itsd);
//        iMPO* cellMPO=o->MakeUnitcelliMPO(L);
//
//        E2=GetExpectation(A,lambdaB(),cellMPO);
//        delete cellMPO;
//    }
//    cout << std::setprecision(9) << "E1,E2=" << E1 -E2 << endl;
    return E1;
}



//
//  We should be able to do these by simply handing off the data inside
//
template <class T> inline Vector<T> Flatten(const Matrix<T>& m)
{
    Vector<T> v(m.size());
    int ij=1;
    for (int j:m.cols())
        for (int i:m.rows())
            v(ij++)=m(i,j);
    return v;
}

template <class T> inline Matrix<T> UnFlatten(const Vector<T>& v,int M, int N)
{
    assert(v.size()==M*N);
    Matrix<T> m(M,N);
    int ij=1;
    for (int j:m.cols())
        for (int i:m.rows())
            m(i,j)=v(ij++);
    return m;
}


Matrix4CT GetTransferMatrix1(const iTEBDStateImp::dVectorT& M)
{
    int d=M.size();
    assert(d>0);
    int D=M[0].GetNumRows();
    assert(D==M[0].GetNumCols());
    Matrix4CT E(D,D,D,D);

    for (int i1=1; i1<=D; i1++)
        for (int j1=1; j1<=D; j1++)
            for (int i3=1; i3<=D; i3++)
                for (int j3=1; j3<=D; j3++)
                {
                    dcmplx e(0);
                    for (int n=0; n<d; n++)
                        e+=conj(M[n](i1,i3))*M[n](j1,j3);
                    E(i1,j1,i3,j3)=e;
                }
    return E;
}
// This code follows: Article (Phien2012) Phien, H. N.; Vidal, G. & McCulloch, I. P.
// Infinite boundary conditions for matrix product state calculations
// Physical Review B, American Physical Society (APS), 2012, 86
//  On entry A[n]=A^na * A^nb ... A^nx = *LambdaA()*Gamma(na) * LambdaB()*Gamma(nb)* ...
//
double iTEBDStateImp::GetExpectation (const dVectorT& A,const DiagonalMatrixRT& lambda, const iMPO* o)
{
    assert(o);
    assert(o->GetL()>=1);
//    o->Report(cout);
//
//  Make sure everything int iMPS is square.
//
    int d=A.size();
    assert(d>0);
    int D=A[0].GetNumRows();
    assert(D==A[0].GetNumCols());

    const SiteOperator* so=o->GetSiteOperator(1);
//    so->Report(cout); cout << endl;
    int Dw=so->GetDw12().Dw1;
    assert(Dw==so->GetDw12().Dw2);
#ifdef DEBUG
//
//  Lets check that W has the correct shape
//
    for (int ma=0; ma<d; ma++)
        for (int na=0; na<d; na++)
        {
            const MatrixRT& Wmn=so->GetW(ma,na);
            assert(IsLowerTriangular(Wmn));
            for (int w1=1;w1<=Dw-1;w1++)
                for (int w2=w1+1;w2<=Dw;w2++)
                    if (Wmn(w1,w2)!=0.0)
                    {
                        cout << "W(" << ma << "," << na << ")=" << Wmn << endl;
                        assert(Wmn(w1,w2)==0.0);
                    }
        }
#endif
//
//  Fill out E[5]
//
    dVectorT E(Dw+1);
    E[Dw]=MatrixCT(D,D);
    Unit(E[Dw]);
//
//  Check E[Dw] is self consisteny
//
    MatrixCT EDw(D,D);
    Fill(EDw,dcmplx(0));
    for (int m=0; m<d; m++)
        for (int n=0; n<d; n++)
        {
            const MatrixRT& Wmn=so->GetW(m,n);
            for (int i2=1;i2<=D;i2++)
            for (int j2=1;j2<=D;j2++)
                for (int i1=1;i1<=D;i1++)
                for (int j1=1;j1<=D;j1++)
                    EDw(i2,j2)+=conj(A[m](i1,i2))*Wmn(Dw,Dw)*E[Dw](i1,j1)*A[n](j1,j2);
        }
//    cout << "E1=" << E1 << endl;
    assert(Max(fabs(E[Dw]-EDw))<1e-12);

//
//  Now loop down from Dw-1=4 down to 2.  Only Row Dw in W is non-zero when column w>1.
//
    MatrixCT T; //Transfer matrix
    for (int w1=Dw-1;w1>=2;w1--)
    {
        std::vector<int> diagonals;
        MatrixCT C(D,D);
        Fill(C,dcmplx(0));
        for (int m=0; m<d; m++)
            for (int n=0; n<d; n++)
            {
                const MatrixRT& Wmn=so->GetW(m,n);
                if (Wmn(w1,w1)!=0.0) //Make sure there is nothing on the diagonal.
                {
                    assert(m==n); //should be only unit ops on the diagonal
                    assert(Wmn(w1,w1)==1.0);
                    diagonals.push_back(w1);
//                    std::cerr << "diagonal W["<< m << "," << n << "](" << w1 << "," << w1 << ")=" << Wmn(w1,w1) << std::endl;
                }
                for (int w2=w1+1;w2<=Dw;w2++)
                if (Wmn(w2,w1)!=0.0)
                {
                    for (int i2=1;i2<=D;i2++)
                    for (int j2=1;j2<=D;j2++)
                        for (int i1=1;i1<=D;i1++)
                        for (int j1=1;j1<=D;j1++)
                            C(i2,j2)+=conj(A[m](i1,i2))*Wmn(w2,w1)*E[w2](i1,j1)*A[n](j1,j2);
                }
            }

        if (diagonals.size()==0)
        {
            E[w1]=C;
//            cout << std::fixed << "E[" << w1 << "]=" << E[w1] << endl;
        }
        else
        {
            Fill(C,dcmplx(0.0));
//            C(1,1)=C(2,2);
//            C(2,2)=C(1,1);
//            cout << std::fixed << "C[" << w1 << "]=" << C << endl;
            dcmplx c=Sum(C.GetDiagonal())/static_cast<double>(D);
            MatrixCT I(D,D);
            Unit(I);
            MatrixCT Cperp=C-c*I;
//            cout << std::fixed << "Cperp[" << w1 << "]=" << Cperp << endl;
            if (T.size()==0)
            {
                T=-GetTransferMatrix1(A).Flatten();
                for (int i=1;i<=D;i++) T(i,i)+=1.0;
//                cout << std::fixed << "1-T=" << T << endl;
            }
//            FillRandom(T);
            LapackSVDSolver<dcmplx> solver;
            auto [U,s,VT]=solver.SolveAll(T,1e-13);
            SVCompressorC* comp =Factory::GetFactory()->MakeMPSCompressor(0,1e-13);
            comp->Compress(U,s,VT);
//            cout << std::fixed << "s=" << s.GetDiagonal() << endl;
            DiagonalMatrixRT si=1.0/s;
//            cout << std::fixed << "si=" << si.GetDiagonal() << endl;
//            MatrixCT err2=U*s*VT-T;
//            cout << "err2=" << std::fixed << err2 << endl;
//            cout << std::scientific << Max(fabs(err2)) << endl;
            MatrixCT V=conj(Transpose(VT));
            MatrixCT UT=conj(Transpose(U));

            MatrixCT Tinv=V*si*UT;
//            MatrixCT err3=T*Tinv*T-T;
//            cout << "err3=" << std::fixed << err3 << endl;
//            cout << std::scientific << Max(fabs(err3)) << endl;


//            cout << "T=" << std::fixed << T << endl;
//            cout << "Tinv=" << std::fixed << Tinv << endl;
            VectorCT Cf=Flatten(C);
            VectorCT Ef=Tinv*Cf;
//            cout << "Cf=" << std::fixed << Cf << endl;
//            cout << "Ef=" << std::fixed << Ef << endl;
//            cout << "T*Ef=" << std::fixed << T*Ef << endl;
            VectorCT err1=T*Ef-Cf;
//            cout << "err1=" << std::fixed << err1 << endl;
//            cout << std::scientific << Max(fabs(err1)) << endl;
            E[w1]=UnFlatten(Ef,D,D);
//            cout << std::fixed << "E[" << w1 << "]=" << E[w1] << endl;
            //
            //  Check solution
            //
            MatrixCT Echeck(D,D);
            Fill(Echeck,dcmplx(0.0));
            for (int i2=1;i2<=D;i2++)
            for (int j2=1;j2<=D;j2++)
                for (int i1=1;i1<=D;i1++)
                for (int j1=1;j1<=D;j1++)
                    for (int m=0; m<d; m++)
                        Echeck(i2,j2)+=conj(A[m](i1,i2))*E[w1](i1,j1)*A[m](j1,j2);
             MatrixCT err=E[w1]-Echeck-C;
//             cout << "err=" << std::fixed << err << endl;
//             cout << std::scientific << Max(fabs(err)) << endl;

        }
    }
//
//  Now do the final contraction to get E[1]
//
    if (Dw>1)
    {
        E[1]=MatrixCT(D,D);
        Fill(E[1],dcmplx(0));
        for (int w=2;w<=Dw;w++)
        {
            for (int m=0; m<d; m++)
                for (int n=0; n<d; n++)
                {
                    const MatrixRT& Wmn=so->GetW(m,n);
                    //cout << "W" << m << n << "=" << Wmn << endl;
                    for (int i2=1;i2<=D;i2++)
                    for (int j2=1;j2<=D;j2++)
                        for (int i1=1;i1<=D;i1++)
                        for (int j1=1;j1<=D;j1++)
                            E[1](i2,j2)+=Wmn(w,1)*conj(A[m](i1,i2))*E[w](i1,j1)*A[n](j1,j2);
                }
        }

    }

//  E[1] should now be the same as C in the paper.
//    cout << "E[" << 1 << "]=" << E[1] << endl;

//
//  Take the trace of E1
//
    dcmplx E0=0.0;
    DiagonalMatrixRT ro=lambda*lambda;
    assert(fabs(Sum(ro)-1.0)<1e-13); //Make sure we are normalized
    for (int i2=1;i2<=D;i2++)
        E0+=E[1](i2,i2)*ro(i2); //We only use the diagonal elements of E[1]
    E0/=2.0; //Convert from energy per unit cell to energy per site.

    if (fabs(imag(E0))>=1e-10)
        Logger->LogWarnV(0,"iTEBDStateImp::GetExpectation(iMPO) E0=(%.5f,%.1e) has large imaginary component",real(E0),imag(E0));
    return real(E0);
}


double iTEBDStateImp::GetExpectationDw1 (const MPO* o) const
{
    int oldCenter=s1.leftSiteNumber;
    double e1=GetExpectationDw1(o,1);
    double e2=GetExpectationDw1(o,2);
    ReCenter(oldCenter);
    return 0.5*(e1+e2);
}

double iTEBDStateImp::GetExpectationDw1 (const MPO* o,int center) const
{
    assert(o->GetL()==2);
    ReCenter(center);
    int D=s1.siteA->GetD1();
    assert(D==s1.siteA->GetD2());
    assert(D==s1.siteB->GetD1());
    assert(D==s1.siteB->GetD2());

    const SiteOperator* soA=o->GetSiteOperator(1);
    const SiteOperator* soB=o->GetSiteOperator(2);
    int DwA2=soA->GetDw12().Dw2;
#ifdef DEBUG
    int DwA1=soA->GetDw12().Dw1;
    int DwB1=soB->GetDw12().Dw1;
    int DwB2=soB->GetDw12().Dw2;
#endif
    assert(DwA1==1);
    assert(DwB2==1);
    assert(DwA2==DwB1);
    dcmplx expectation1(0.0);
    for (int na=0; na<itsd; na++)
        for (int nb=0; nb<itsd; nb++)
        {
            MatrixCT theta13_n=lambdaB()*GammaA()[na]*lambdaA()*GammaB()[nb]*lambdaB();
            assert(theta13_n.GetNumRows()==D);
            assert(theta13_n.GetNumCols()==D);
            for (int ma=0; ma<itsd; ma++)
                for (int mb=0; mb<itsd; mb++)
                {
                    const MatrixRT& WAmn=soA->GetW(ma,na);
                    const MatrixRT& WBmn=soB->GetW(mb,nb);
                    assert(WAmn.GetNumRows()==1);
                    assert(WAmn.GetNumCols()==DwA2);
                    assert(WBmn.GetNumRows()==DwA2);
                    assert(WBmn.GetNumCols()==1);
                    double Omn(0);
                    for (int w2=1; w2<=DwA2; w2++)
                        Omn+=WAmn(1,w2)*WBmn(w2,1);

                    MatrixCT theta13_m=conj(lambdaB()*GammaA()[ma])*lambdaA()*conj(GammaB()[mb])*lambdaB();
                    assert(theta13_m.GetNumRows()==D);
                    assert(theta13_m.GetNumCols()==D);
                    for (int i1=1; i1<=D; i1++)
                        for (int i3=1; i3<=D; i3++)
                            expectation1+=theta13_m(i1,i3)*Omn*theta13_n(i1,i3);
                }
        }

    if (fabs(imag(expectation1))>=1e-10)
        Logger->LogWarnV(0,"iTEBDStateImp::GetExpectation expectation=(%.5f,%.1e) has large imaginary component",real(expectation1),imag(expectation1));
    return real(expectation1);
}

//double iTEBDStateImp::GetExpectation (const MPO* o) const
//{
//    const MPSSite* first=itsSites[1];
//    Matrix4CT E=first->GetTransferMatrix(DLeft);
//    EigenSolver<dcmplx>* solver=new ArpackEigenSolver<dcmplx>;
//
//    int D1=first->GetD1();
//    const Dw12& DWs=o->GetSiteOperator(1)->GetDw12();
//    Vector3CT F(DWs.Dw1,D1,D1,1);
//    {
//        auto [U,d]=solver->SolveLeft_NonSym(E.Flatten(),1e-13,1);
//        F=U.GetColumn(1);
//        cout << std::fixed << std::setprecision(5) << "left eigen value=" << d(1) << endl;
//
//        index_t ij=1;
//        for (index_t j=1;j<=D1;j++)
//            for (index_t i=1;i<=D1;i++,ij++)
//                assert(F(1,i,j)==U.GetColumn(1)(ij));
//    }
//    SiteLoop(ia)
//        F=itsSites[ia]->IterateLeft_F(o->GetSiteOperator(ia),F);
//
//    {
//
//    }
//    const MPSSite* last=itsSites[itsL];
//    E=last->GetTransferMatrix(DRight);
//    auto [U,d]=solver->SolveRightNonSym(E.Flatten(),1e-13,1);
//    cout << std::fixed << std::setprecision(5) << "right eigen value=" << d(1) << endl;
//    dcmplx ret=0.0;//F.Flatten()*U.GetColumn(1);
//    index_t ij=1;
//    for (index_t j=1;j<=D1;j++)
//        for (index_t i=1;i<=D1;i++,ij++)
//            ret+=F(1,j,i)*U.GetColumn(1)(ij);
//    cout << "ret=" << ret << endl;
//
//    double ir=std::imag(ret)/itsL/itsL;
//    if (fabs(ir)>1e-10)
//        cout << "Warning: MatrixProductState::GetExpectation Imag(E)=" << ir << endl;
//    delete solver;
//
//    return std::real(ret);
//}


}
