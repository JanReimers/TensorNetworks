#include "TensorNetworksImp/iTEBDStateImp.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/TNSLogger.H"

#include "TensorNetworksImp/Bond.H"

#include "NumericalMethods/ArpackEigenSolver.H"
#include "NumericalMethods/PrimeEigenSolver.H"
#include "NumericalMethods/LapackEigenSolver.H"
#include "NumericalMethods/LapackSVD.H"
#include "oml/cnumeric.h"
#include <iomanip>

namespace TensorNetworks
{

MPSSite::dVectorT  iTEBDStateImp::ContractTheta(const Matrix4RT& expH) const
{
    //
    //  Make sure everything is square
    //
    assert(s1.siteA->GetD2()==s1.siteB->GetD1());
    assert(s1.siteA->GetD1()==s1.siteB->GetD2());
    assert(s1.siteA->GetD1()==s1.siteA->GetD2());
    int D=s1.siteA->GetD1();
    //
    //  Create empty theta tensor
    //
    dVectorT  Thetap(itsd*itsd);
    for (int n=0;n<itsd*itsd;n++)
    {
        Thetap[n].SetLimits(D,D);
        Fill(Thetap[n],dcmplx(0.0));
    }
    //
    //  Contract
    //
    for (int mb=0; mb<itsd; mb++)
    for (int ma=0; ma<itsd; ma++)
    {
        MatrixCT theta13  =GammaA()[ma]*lambdaA()*GammaB()[mb];  //Figure 14 v
        int nab=0;
        for (int nb=0; nb<itsd; nb++)
        for (int na=0; na<itsd; na++,nab++)
            Thetap[nab]+= theta13*expH(ma,na,mb,nb);
    }

    return Thetap;
}

MPSSite::dVectorT  iTEBDStateImp::ContractTheta(const MPO* o) const
{
    //
    //  Make sure everything is square
    //
    assert(s1.siteA->GetD2()==s1.siteB->GetD1());
    assert(s1.siteA->GetD1()==s1.siteB->GetD2());
    assert(s1.siteA->GetD1()==s1.siteA->GetD2());
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
    //  Here we choose the contract the core part of the diagram and add on the lambdaB's
    //  later, as needed.  When Dw===w1==w3==1 (i.e. open legs at the edges of the operator) then this is
    //  easy.  But when Dw > 1 then this become non-trivial because the indices get combined (i1,w1) (i3,w3).
    //  One solution is to simply make all three versions listed above in this routine.  The other is to extend
    //  lambdaB with Dw copies of itself.
    //
    for (int mb=0; mb<itsd; mb++)
    for (int ma=0; ma<itsd; ma++)
    {
        MatrixCT theta13  =GammaA()[ma]*lambdaA()*GammaB()[mb];  //Figure 14 v
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
