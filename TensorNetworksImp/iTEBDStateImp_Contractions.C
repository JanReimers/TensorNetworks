#include "TensorNetworksImp/iTEBDStateImp.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/TNSLogger.H"

#include "TensorNetworksImp/Bond.H"

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

MPSSite::dVectorT  iTEBDStateImp::ContractTheta(const MPO* o,ThetaType tt) const
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

MPSSite::dVectorT  iTEBDStateImp::ContractThetaDw(const MPO* o) const
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
    double E1,E2;
    ReCenter(1);
    {
        cout << "oerr=" << GetOrthonormalityErrors() << endl;
        assert(GetOrthonormalityErrors()<1e-11);

        dVectorT gamma=lambdaB()*ContractAlB();
        assert(gamma.size()==itsd*itsd);
        iMPO* cellMPO=o->MakeUnitcelliMPO(L);

        E1=GetExpectation(gamma,cellMPO);
        delete cellMPO;
    }
    ReCenter(2);
    {
        cout << "oerr=" << GetOrthonormalityErrors() << endl;
        assert(GetOrthonormalityErrors()<1e-11);

        dVectorT gamma=lambdaB()*ContractAlB();
        assert(gamma.size()==itsd*itsd);
        iMPO* cellMPO=o->MakeUnitcelliMPO(L);

        E2=GetExpectation(gamma,cellMPO);
        delete cellMPO;
    }

    return (E1+E2)/sqrt(1.0);
}


// This code follows: McCulloch, I. P. Infinite size density matrix renormalization group, revisited
//                    http://arxiv.org/pdf/0804.2509v1
//  On entry gamma[n]=A^na * A^nb ... A^nx = Gamma(na)*LambdaB() *  Gamma(nb)*LambdaC() ...  Gamma(nx)*LambdaA()
//
double iTEBDStateImp::GetExpectation (const dVectorT& gamma, const iMPO* o)
{
    assert(o);
    assert(o->GetL()>=1);
    o->Report(cout);
//
//  Make sure everything int iMPS is square.
//
    int d=gamma.size();
    assert(d>0);
    int D=gamma[0].GetNumRows();
    assert(D==gamma[0].GetNumCols());

    const SiteOperator* so=o->GetSiteOperator(1);
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
            for (int w1=1;w1<=Dw-1;w1++)
                for (int w2=2;w2<=Dw;w2++)
                    assert(Wmn(w1,w2)==0.0);
        }
#endif
//
//  Fill out E[1]
//
    dVectorT E(Dw+1);
    E[1]=MatrixCT(D,D);
    Unit(E[1]);
//
//  Check E[1] is self consisteny
//
    MatrixCT E1(D,D);
    Fill(E1,dcmplx(0));
    for (int m=0; m<d; m++)
        for (int n=0; n<d; n++)
        {
            const MatrixRT& Wmn=so->GetW(m,n);
            for (int i2=1;i2<=D;i2++)
            for (int j2=1;j2<=D;j2++)
                for (int i1=1;i1<=D;i1++)
                    E1(i2,j2)+=conj(gamma[m](i1,i2))*Wmn(1,1)*gamma[n](i1,j2);
        }
//    cout << "E1=" << E1 << endl;
    assert(Max(fabs(E[1]-E1))<1e-12);

//
//  Now loop down from 2 to Dw-1.  Only column 1 in W is non-zero.
//
    for (int w=2;w<=Dw-1;w++)
    {
        E[w]=MatrixCT(D,D);
        Fill(E[w],dcmplx(0));
        for (int m=0; m<d; m++)
            for (int n=0; n<d; n++)
            {
                const MatrixRT& Wmn=so->GetW(m,n);
                for (int i2=1;i2<=D;i2++)
                for (int j2=1;j2<=D;j2++)
                    for (int i1=1;i1<=D;i1++)
                        E[w](i2,j2)+=conj(gamma[m](i1,i2))*Wmn(w,1)*gamma[n](i1,j2);
            }
//        cout << "E[" << w << "]=" << E[w] << endl;
    }
//
//  Now do the final contraction to get E[1]
//
    if (Dw>1)
    {
        E[Dw]=MatrixCT(D,D);
        Fill(E[Dw],dcmplx(0));
        for (int w=1;w<=Dw-1;w++)
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
                            E[Dw](i2,j2)+=Wmn(Dw,w)*conj(gamma[m](i1,i2))*E[w](i1,j1)*gamma[n](j1,j2);
                }
        }

    }
//    cout << "E[" << Dw << "]=" << E[Dw] << endl;

//
//  Take the trace of E1
//
    dcmplx E0=0.0;
    for (int i2=1;i2<=D;i2++)
    for (int j2=1;j2<=D;j2++)
        E0+=E[Dw](i2,j2);
//    cout << "E0=" << E0 << endl;
    E0/=D;
    E0/=D;
//    cout << "E0=" << E0 << endl;
    if (fabs(imag(E0))>=1e-10)
        Logger->LogWarnV(0,"iTEBDStateImp::GetExpectation(iMPO) E0=(%.5f,%.1e) has large imaginary component",real(E0),imag(E0));
    return real(E0);
}

double iTEBDStateImp::GetExpectation (const MPO* o) const
{
    int oldCenter=s1.leftSiteNumber;
    double e1=GetExpectation(o,1);
    double e2=GetExpectation(o,2);
    ReCenter(oldCenter);
    return 0.5*(e1+e2);
}

double iTEBDStateImp::GetExpectation (const MPO* o,int center) const
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
