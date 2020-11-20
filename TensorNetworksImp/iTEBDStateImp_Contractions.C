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

Matrix4CT iTEBDStateImp::GetTransferMatrix(const dVectorT& M) const
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

MatrixCT  iTEBDStateImp::GetNormMatrix(Direction lr,const dVectorT& M) const //Er*I or I*El
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
//  Assume two site for now
//
Matrix4CT iTEBDStateImp::GetTransferMatrix(Direction lr) const
{
    assert(s1.siteA->GetD1()==s1.siteA->GetD2());
    assert(s1.siteA->GetD1()==s1.siteB->GetD1());
    assert(s1.siteA->GetD1()==s1.siteB->GetD2());

    dVectorT gamma(itsd*itsd);
    int nab=0;
    for (int nb=0; nb<itsd; nb++)
        for (int na=0; na<itsd; na++,nab++)
            gamma[nab]=GammaA()[na]*lambdaA()*GammaB()[nb];
    Matrix4CT E;
    switch (lr)
    {
    case DRight :
        E=GetTransferMatrix(gamma*lambdaB());
        break;
    case DLeft :
        E=GetTransferMatrix(lambdaB()*gamma);
        break;
    }

    return E;
}


Matrix4CT iTEBDStateImp::GetTransferMatrix(const Matrix4CT& theta) const
{
    int D=s1.siteA->GetD1();
    Matrix4CT E(D,D,D,D);

    for (int i1=1; i1<=D; i1++)
        for (int j1=1; j1<=D; j1++)
            for (int i3=1; i3<=D; i3++)
                for (int j3=1; j3<=D; j3++)
                {
                    dcmplx e(0);
                    for (int na=1; na<=itsd; na++)
                    for (int nb=1; nb<=itsd; nb++)
                        e+=conj(theta(na,j1,nb,j3))*(theta(na,i1,nb,i3));
                    E(i1,j1,i3,j3)=e;
                }
    return E;
}
double iTEBDStateImp::GetExpectationmmnn (const Matrix4RT& Hlocal) const
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
