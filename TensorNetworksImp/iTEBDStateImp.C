#include "TensorNetworksImp/iTEBDStateImp.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "NumericalMethods/ArpackEigenSolver.H"
#include "NumericalMethods/PrimeEigenSolver.H"
#include "NumericalMethods/LapackEigenSolver.H"
#include "NumericalMethods/LapackSVD.H"
#include "oml/cnumeric.h"
#include <iomanip>

namespace TensorNetworks
{

iTEBDStateImp::iTEBDStateImp(int L,double S, int D,double normEps,double epsSV)
    : MPSImp(L,S,DLeft,normEps)
{
    InitSitesAndBonds(D,epsSV);
}

iTEBDStateImp::~iTEBDStateImp()
{
    itsBonds[0]=0; //Avoid double deletion in optr_vector destructor
    //dtor
}


void iTEBDStateImp::InitSitesAndBonds(int D,double epsSV)
{
    //
    //  Create bond objects
    //
    itsBonds.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    for (int i=1; i<=itsL; i++)
        itsBonds.push_back(new Bond(epsSV));
    itsBonds[0]=itsBonds[itsL];  //Periodic boundary conditions
    //
    //  Create Sites
    //
    itsSites.push_back(0);  //Dummy space holder. We want this array to be 1 based.
    for (int i=1; i<=itsL-1; i++)
        itsSites.push_back(new MPSSite(PBulk,itsBonds[i-1],itsBonds[i],itsd,D,D));
    itsSites.push_back(new MPSSite(PRight,itsBonds[itsL-1],itsBonds[0],itsd,D,D));
    //
    //  Tell each bond about its left and right sites.
    //
    for (int i=1; i<=itsL; i++)
        itsBonds[i]->SetSites(itsSites[i],itsSites[GetModSite(i+1)]);
}

void iTEBDStateImp::InitializeWith(State s)
{
    MPSImp::InitializeWith(s);
}



void iTEBDStateImp::Normalize(Direction lr)
{
    Matrix4CT E=GetTransferMatrix();
    EigenSolver<dcmplx>* solver=new ArpackEigenSolver<dcmplx>;
    dcmplx left_eigenValue(0);
    {
        auto [U,d]=solver->SolveLeft_NonSym(E.Flatten(),1e-13,1);
//        cout << std::fixed << std::setprecision(4) << "Left  Arpack d=" << d << endl;
        left_eigenValue=d(1);
    }
//    dcmplx right_eigenValue(0);
//    {
//        auto [U,d]=solver->SolveRightNonSym(E.Flatten(),1e-13,1);
//        cout << std::fixed << std::setprecision(4) << "Right Arpack d=" << d << endl;
//        right_eigenValue=d(1);
//    }
//    delete solver;
    assert(fabs(imag(left_eigenValue))<1e-10);
//    assert(fabs(imag(right_eigenValue))<1e-10);
    double lnorm=sqrt(real(left_eigenValue));
//    double rnorm=sqrt(real(right_eigenValue));
    double fa=itsSites[1]->FrobeniusNorm();
    double fb=itsSites[2]->FrobeniusNorm();
    cout << "lnorm, fa,fb=" << lnorm << " " << fa << " " << fb << endl;
    itsSites[1]->Rescale(sqrt(lnorm));
    itsSites[2]->Rescale(sqrt(lnorm));

}

void iTEBDStateImp::Canonicalize(Direction lr)
{
    ForLoop(lr)
    MPSImp::CanonicalizeSite(lr,ia,0);
}

//void iTEBDStateImp::NormalizeAndCompress(Direction LR,int Dmax,double epsMin);
int iTEBDStateImp::GetModSite(int isite) const
{
    int modSite=((isite-1)%itsL)+1;
    assert(modSite>=1);
    assert(modSite<=itsL);
    return modSite;
}

const DiagonalMatrixRT&  iTEBDStateImp::GetLambda(int isite) const
{
    const MPSSite* site=itsSites[GetModSite(isite  )];
    assert(site);
    const Bond*    bond=site->itsRightBond;
    assert(bond);
    return bond->itsSingularValues;
}

const MatrixCT& iTEBDStateImp::GetGamma (int isite,int n) const
{
    assert(n>=0);
    assert(n<itsd);
    const MPSSite* site=itsSites[GetModSite(isite  )];
    assert(site);
    return site->itsMs[n];
}

iTEBDStateImp::Sites::Sites(int leftSite, iTEBDStateImp* iTEBD)
    : siteA(iTEBD->itsSites[iTEBD->GetModSite(leftSite  )])
    , siteB(iTEBD->itsSites[iTEBD->GetModSite(leftSite+1)])
    , bondA(siteA->itsRightBond)
    , bondB(siteB->itsRightBond)
    , MA(siteA->itsMs)
    , MB(siteB->itsMs)
    , lambdaA(bondA->GetSVs())
    , lambdaB(bondB->GetSVs())
{

}

MatrixCT ReshapeForSVD(int d,const MPSSite::dVectorT& M)
{
    assert(d>0);
    assert(M.size()==d*d);
    int D=M[0].GetNumRows();
    assert(D==M[0].GetNumCols());
    Matrix4CT ret(D,d,D,d);
    int nab=0;
    for (int nb=0; nb<d; nb++)
    for (int na=0; na<d; na++,nab++)
        for (int j=1; j<=D; j++)
        for (int i=1; i<=D; i++)
        {
            ret(i,na+1,j,nb+1)=M[na+d*nb](i,j);
        }

    return ret.Flatten();
}

void iTEBDStateImp::Orthogonalize(int isite)
{
    Sites s(isite,this);
    dVectorT gamma(itsd*itsd);
    int nab=0;
    for (int nb=0; nb<itsd; nb++)
        for (int na=0; na<itsd; na++,nab++)
            gamma[na+itsd*nb]=s.MA[na]*s.lambdaA*s.MB[nb];
    auto [gammap,lambdap]=Orthogonalize(gamma,s.lambdaB);

    s.bondB->SetSingularValues(lambdap,0.0);
    DiagonalMatrixRT lbinv=1.0/lambdap; //inverse of LambdaB
//    cout << "lambdap* 1/lambdap=" << lambdap*lbinv << endl;
    //
    //  Now unpack gamma' into gammaA'*lambdaA'*gammaB'
    //
    dVectorT bgb(itsd*itsd);
    for (int n=0; n<itsd*itsd; n++)
        bgb[n]=lambdap*gammap[n]*lambdap; // Sandwich LambdaB*gammap*LambdaB

    MatrixCT bgb4=ReshapeForSVD(itsd,bgb);

    int D=s.lambdaA.size();
    SVDSolver<dcmplx>* svd_solver=new LapackSVDSolver<dcmplx>();
    auto [P,lambdaA_prime,Q]=svd_solver->Solve(bgb4,1e-13,D); //only keep D svs.
    assert(Max(fabs(P*lambdaA_prime*Q-bgb4))<1e-13);
    s.bondA->SetSingularValues(lambdaA_prime,0.0);
    cout << std::fixed << "lambdaB_prime" << lambdap.GetDiagonal() << endl;
    cout << std::fixed << "lambdaA_prime" << lambdaA_prime.GetDiagonal() << endl;
//    cout << "P=" << P << endl;
//    cout << "Q=" << Q << endl;
    assert(P.GetNumCols()==D);
    assert(P.GetNumRows()==D*itsd);
    assert(Q.GetNumCols()==D*itsd);
    assert(Q.GetNumRows()==D);
    for (int n=0; n<itsd; n++)
    {
        for (int i=1; i<=D; i++)
        for (int j=1; j<=D; j++)
        {
//            cout << n << " " << i << " " << j << " " << n*D+i << " " << n*D+j << endl;
            s.MA[n](i,j)=lbinv(i)*P(n*D+i,       j);
            s.MB[n](i,j)=         Q(       i,n*D+j)*lbinv(j);
        }
//        s.MA[n]=lbinv*s.MA[n];
//        s.MB[n]=s.MB[n]*lbinv;
//        cout << "n,GA,GB=" << n << " " << s.MA[n] << " " << s.MB[n]<< endl;
    }

    delete svd_solver;

//    dVectorT gammap1(itsd*itsd);
//    nab=0;
//    for (int nb=0; nb<itsd; nb++)
//        for (int na=0; na<itsd; na++,nab++)
//            gammap1[na+itsd*nb]=s.MA[na]*lambdaA_prime*s.MB[nb];
//
//    for (int n=0;n<itsd*itsd;n++)
//        cout << "gammap1-gammap=" << Max(fabs(gammap1[n] - gammap[n])) << endl;


    assert(TestOrthogonal(1));
}

MPSSite::dVectorT operator*(const MPSSite::dVectorT& gamma, const DiagonalMatrixRT& lambda)
{
    int d=gamma.size();
    MPSSite::dVectorT gl(d);
    for (int n=0; n<d; n++)
        gl[n]=gamma[n]*lambda;
    return gl;
}
MPSSite::dVectorT operator*(const DiagonalMatrixRT& lambda, const MPSSite::dVectorT& gamma)
{
    int d=gamma.size();
    MPSSite::dVectorT lg(d);
    for (int n=0; n<d; n++)
        lg[n]=lambda*gamma[n];
    return lg;
}

MatrixCT operator*(const Matrix4CT& E, const MatrixCT& Vr)
{
    int D=Vr.GetNumRows();
    MatrixCT Evr(D,D);
    for (int i1=1; i1<=D; i1++)
    for (int j1=1; j1<=D; j1++)
    {
        dcmplx evr(0);
        for (int i2=1; i2<=D; i2++)
        for (int j2=1; j2<=D; j2++)
            evr+=E(i1,j1,i2,j2)*Vr(i2,j2);
        Evr(i1,j1)=evr;
    }
    return Evr;
}
MatrixCT operator*(const MatrixCT& Vl,const Matrix4CT& E)
{
    int D=Vl.GetNumRows();
    MatrixCT Evl(D,D);
    for (int i2=1; i2<=D; i2++)
    for (int j2=1; j2<=D; j2++)
    {
        dcmplx evl(0);
        for (int i1=1; i1<=D; i1++)
        for (int j1=1; j1<=D; j1++)
            evl+=Vl(i1,j1)*E(i1,j1,i2,j2);
        Evl(i2,j2)=evl;
    }
    return Evl;
}

bool iTEBDStateImp::TestOrthogonal(int isite)
{
    Sites s(isite,this);
    dVectorT gamma(itsd*itsd);
    int D=s.lambdaA.size();
    int nab=0;
    for (int na=0; na<itsd; na++)
        for (int nb=0; nb<itsd; nb++,nab++)
            gamma[nab]=s.MA[na]*s.lambdaA*s.MB[nb];
    Matrix4CT Er=GetTransferMatrix(gamma*s.lambdaB);
    Matrix4CT El=GetTransferMatrix(s.lambdaB*gamma);
    MatrixCT I(D,D); //Right ei
    Unit(I);
    MatrixCT ErI=Er*I;
    MatrixCT IEl=I*El;
    cout << std::scientific;
    cout << "Er*I-I=" << Max(fabs(ErI-I)) << endl;
    cout << "i*El-I=" << Max(fabs(IEl-I)) << endl;
//    cout << "IsUnit(ErI,1e-13)" << IsUnit(ErI,1e-12) << endl;
//    cout << "IsUnit(IEl,1e-13)" << IsUnit(IEl,1e-12) << endl;
    return IsUnit(ErI,1e-9) && IsUnit(IEl,1e-9);
}

iTEBDStateImp::GLType iTEBDStateImp::Orthogonalize(const dVectorT& gamma, const DiagonalMatrixRT& lambda)
{
    int d=gamma.size();
    assert(d>0);
    int D=gamma[0].GetNumRows();
    assert(D==gamma[0].GetNumCols());

    //
    //  Calculate right and left transfer matrices and the R/L eigen vectors
    //  Transform the eigen vectors into Hermitian eigen matrices
    //
    EigenSolver<dcmplx>* solver=new ArpackEigenSolver<dcmplx>;

    dVectorT gl=gamma*lambda;
    for (int n=0;n<d;n++)
        assert(gl[n]==gamma[n]*lambda);
    Matrix4CT Er=GetTransferMatrix(gamma*lambda);
//    cout << std::fixed << "Er=" << Er << endl;

    MatrixCT Vr(D,D); //Right eigen matrix
    {
        auto [U,e]=solver->SolveRightNonSym(Er.Flatten(),1e-13,1);
//        cout << "Right e 1-e=" << e << " " << 1.0-e(1) << endl;
        assert(fabs(1.0-e(1))<1e-10);
        int ij=1;
        for (int j=1; j<=D; j++)
            for (int i=1; i<=D; i++,ij++)
                Vr(i,j)=U.GetColumn(1)(ij);
        dcmplx phase=Vr(1,1)/fabs(Vr(1,1));
        Vr*=conj(phase); //Take out arbitrary phase angle
    }
//    cout << "Vr=" << Vr << endl;
    MatrixCT rerr=Er*Vr-Vr;
//    cout << "rerr=" << rerr << endl;
    assert(Max(fabs(rerr))<1e-10);

    Matrix4CT El=GetTransferMatrix(lambda*gamma);
    MatrixCT Vl(D,D); //Right eigen matrix
    {
        auto [U,e]=solver->SolveLeft_NonSym(El.Flatten(),1e-13,1);
//        cout << "Left e 1-e=" << e << " " << 1.0-e(1) << endl;
        assert(fabs(1.0-e(1))<1e-10);
        int ij=1;
        for (int j=1; j<=D; j++)
            for (int i=1; i<=D; i++,ij++)
                Vl(i,j)=U.GetColumn(1)(ij);
        dcmplx phase=Vl(1,1)/fabs(Vl(1,1));
        Vl*=conj(phase); //Take out arbitrary phase angle
    }
//    cout << "Vl=" << Vl << endl;
    MatrixCT lerr=Vl*El-Vl;
    assert(Max(fabs(lerr))<1e-10);
//    cout << "lerr=" << lerr << endl;
    delete solver;
//
//  Decompose eigen matrices
//
    solver=new LapackEigenSolver<dcmplx>; //Switch to a dense solver
    assert(IsHermitian(Vr,1e-10));
    assert(IsHermitian(Vl,1e-10));
    MatrixCT X,Xinv;
    {
        auto [U,e]=solver->SolveAll(Vr,1e-13);
        X=U*DiagonalMatrix<double>(sqrt(e));
        Xinv=DiagonalMatrix<double>(1.0/sqrt(e))*Transpose(conj(U));
    }
    assert(IsUnit(X*Xinv,1e-13));
    MatrixCT YT,YTinv;
    {
        auto [U,e]=solver->SolveAll(Vl,1e-13);
        YT=Transpose(U*DiagonalMatrix<double>(sqrt(e)));
        YTinv=Transpose(DiagonalMatrix<double>(1.0/sqrt(e))*Transpose(conj(U)));
    }
    assert(IsUnit(YT*YTinv,1e-13));
    delete solver;
    //
    //  Transform lambda
    //
    MatrixCT YlX=YT*lambda*X;
    SVDSolver<dcmplx>* svd_solver=new LapackSVDSolver<dcmplx>();
    auto [U,lambda_prime,VT]=svd_solver->SolveAll(YlX,1e-13);
    delete svd_solver;
    //
    //  Transform Gamma
    //
    dVectorT gamma_prime(d);
    for (int n=0;n<d;n++)
        gamma_prime[n]=VT*Xinv*gamma[n]*YTinv*U;
    //
    //  Verify orthogonaly
    //
    Er=GetTransferMatrix(gamma_prime*lambda_prime);
    El=GetTransferMatrix(lambda_prime*gamma_prime);
    MatrixCT I(D,D); //Right ei
    Unit(I);
    MatrixCT ErI=Er*I;
    MatrixCT IEl=I*El;
//    cout << std::scientific << "Er*I=" << ErI-I << endl;
//    cout << "i*El=" << IEl-I << endl;
    assert(IsUnit(ErI,1e-9));
    assert(IsUnit(IEl,1e-9));

    return std::make_tuple(gamma_prime,lambda_prime);
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

//
//  Assume two site for now
//
Matrix4CT iTEBDStateImp::GetTransferMatrix() const
{
    MPSSite* siteA=itsSites[GetModSite(1)];
    MPSSite* siteB=itsSites[GetModSite(2)];
    Bond*    bondA=siteA->itsRightBond;
    Bond*    bondB=siteB->itsRightBond;
    assert(siteA);
    assert(siteB);
    MPSSite::dVectorT& MA(siteA->itsMs);
    MPSSite::dVectorT& MB(siteB->itsMs);
    const DiagonalMatrixRT& lambdaA=bondA->GetSVs();
    const DiagonalMatrixRT& lambdaB=bondB->GetSVs();

    int D=siteA->GetD1();
    assert(D==siteA->GetD2());
    assert(D==siteB->GetD1());
    assert(D==siteB->GetD2());

    Matrix4CT E(D,D,D,D);
    E.Fill(0);
    for (int na=0; na<itsd; na++)
        for (int nb=0; nb<itsd; nb++)
        {
            MatrixCT theta13=MA[na]*lambdaA*MB[nb]*lambdaB;
            MatrixCT theta13c=conj(MA[na])*lambdaA*conj(MB[nb])*lambdaB;
            assert(conj(theta13c)==theta13);
            assert(theta13.GetNumRows()==D);
            assert(theta13.GetNumCols()==D);
            for (int i1=1; i1<=D; i1++)
                for (int j1=1; j1<=D; j1++)
                    for (int i3=1; i3<=D; i3++)
                        for (int j3=1; j3<=D; j3++)
                            E(i1,j1,i3,j3)+=theta13c(i1,i3)*theta13(j1,j3);
        }
    return E;
}

void iTEBDStateImp::Apply(int isite,const Matrix4RT& expH,SVCompressorC* comp)
{
    assert(comp);
    MPSSite* siteA=itsSites[GetModSite(isite  )];
    Bond*    bondA=siteA->itsRightBond;
    MPSSite* siteB=itsSites[GetModSite(isite+1)];
    Bond*    bondB=siteB->itsRightBond;
    assert(siteA);
    assert(siteB);
    assert(bondA);
    assert(bondB);
    assert(siteA!=siteB);
    assert(bondA!=bondB);
    MPSSite::dVectorT& MA(siteA->itsMs);
    MPSSite::dVectorT& MB(siteB->itsMs);
    const DiagonalMatrixRT& lambdaA=bondA->GetSVs();
    const DiagonalMatrixRT& lambdaB=bondB->GetSVs();
    //
    //  New we need to contract   Theta(nA,i1,nB,i3) =
    //                         sB(i1)*MA(mA,i1,i2)*sA(i2)*MB(mB,i2,i3)*sB(i3)
    //                                   |                   |
    //                              expH(mA,nA,              mB,nB)
    //
    //  Make sure everything is square
    assert(siteA->GetD2()==siteB->GetD1());
    assert(siteA->GetD1()==siteB->GetD2());
    assert(siteA->GetD1()==siteA->GetD2());
    int D=siteA->GetD1();
    Matrix4CT Theta(itsd,D,itsd,D);
    Theta.Fill(0.0);

    for (int ma=0; ma<itsd; ma++)
        for (int mb=0; mb<itsd; mb++)
        {
            MatrixCT theta13=lambdaB*MA[ma]*lambdaA*MB[mb]*lambdaB;
            for (int na=0; na<itsd; na++)
                for (int nb=0; nb<itsd; nb++)
                    for (int i1=1; i1<=D; i1++)
                        for (int i3=1; i3<=D; i3++)
                            Theta(na+1,i1,nb+1,i3)+=theta13(i1,i3)*expH(ma,na,mb,nb);
//                for (int i1=1;i1<=D;i1++)
//                for (int i3=1;i3<=D;i3++)
//                    Theta(ma+1,i1,mb+1,i3)+=theta13(i1,i3)  ;

        }


    //
    //  Now SVD Theta
    //
    MatrixCT ThetaF=Theta.Flatten();
    assert(ThetaF.GetNumRows()==ThetaF.GetNumCols());
//    cout << std::fixed << std::setprecision(3) << "Theta=" << Theta << endl;
//    cout << "Theta=" << ThetaF.GetLimits() << endl;
//    cout << "Before Compress Dw1 Dw2 A=" << itsDw12.Dw1 << " " << itsDw12.Dw2 << " "<< A << endl;
    auto [U,s,Vdagger]=oml_CSVDecomp(ThetaF);
//    cout << std::fixed << std::setprecision(3) << "Before comp s=" << s.GetDiagonal() << endl;
    double sums1=Sum(s*s);
    assert(Max(fabs(U*s*Vdagger-ThetaF))<1e-10);
//    cout << "before compression U=" << U.GetLimits() << endl;
//    cout << "Vdagger=" << Vdagger << endl;
    double integratedS2=comp->Compress(U,s,Vdagger);
    double sums2=Sum(s*s);
//    cout << std::fixed << std::setprecision(3) << "After comp s=" << s.GetDiagonal() << endl;
    cout << "sums1,sums2,s2/s1=" << sums1 << " " << sums2 << " " << sums2/sums1 << endl;
//    cout << "After compression U=" << U.GetLimits() << endl;
    s/=sqrt(sums2);
    bondA->SetSingularValues(s,integratedS2); //No SVs were thrown away (yet!)

    int nai1=1;
    for (int i1=1; i1<=D; i1++)
        for (int na=0; na<itsd; na++,nai1++)
            for (int i2=1; i2<=D; i2++)
            {
                assert(lambdaB(i1,i1)>1e-10);
//                cout << na << " " << i1 << " " << i2 << " " << nai1 << " " << lambdaB(i1,i1) << endl;
                MA[na](i1,i2)=U(nai1,i2)/lambdaB(i1,i1);
            }
//    cout << "-----------------------------------------------------------------------------" << endl;
    int nbi2=1;
    for (int i3=1; i3<=D; i3++)
        for (int nb=0; nb<itsd; nb++,nbi2++)
            for (int i2=1; i2<=D; i2++)
            {
                assert(lambdaB(i3,i3)>1e-10);
//                cout << nb << " " << i2 << " " << i3 << " " << nbi2 << " " << lambdaB(i3,i3) << endl;
                MB[nb](i2,i3)=Vdagger(i2,nbi2)/lambdaB(i3,i3);
            }

//    cout << "-----------------------------------------------------------------------------" << endl;
//    if (isite==1)
//        siteA->iNormalize(DLeft);
//    if (isite==2)
//        siteA->iNormalize(DRight);

}

double iTEBDStateImp::GetExpectation (int isite, const Matrix4RT& Hlocal) const
{
    Sites s(isite,const_cast<iTEBDStateImp*>(this));

    int D=s.siteA->GetD1();
    assert(D==s.siteA->GetD2());
    assert(D==s.siteB->GetD1());
    assert(D==s.siteB->GetD2());

    Matrix4CT Eo(D,D,D,D);
    Eo.Fill(0);
    for (int na=0; na<itsd; na++)
        for (int nb=0; nb<itsd; nb++)
        {
            MatrixCT theta13_n=s.MA[na]*s.lambdaA*s.MB[nb]*s.lambdaB;
            assert(theta13_n.GetNumRows()==D);
            assert(theta13_n.GetNumCols()==D);
            for (int ma=0; ma<itsd; ma++)
                for (int mb=0; mb<itsd; mb++)
                {
                    MatrixCT theta13_m=conj(s.MA[ma])*s.lambdaA*conj(s.MB[mb])*s.lambdaB;
                    assert(theta13_m.GetNumRows()==D);
                    assert(theta13_m.GetNumCols()==D);
                    for (int i1=1; i1<=D; i1++)
                        for (int j1=1; j1<=D; j1++)
                            for (int i3=1; i3<=D; i3++)
                                for (int j3=1; j3<=D; j3++)
                                    Eo(i1,j1,i3,j3)+=theta13_m(i1,i3)*Hlocal(ma,mb,na,nb)*theta13_n(j1,j3);
                }
        }
//    cout << std::fixed << std::setprecision(4) << "Eo=" << Eo << endl;
    Matrix4CT E=GetTransferMatrix();
//    cout << std::fixed << std::setprecision(4) << "E=" << E << endl;


    EigenSolver<dcmplx>* solver=new ArpackEigenSolver<dcmplx>;
    VectorCT Left_EigenVector;
    dcmplx eigenValue(0);
    {
        auto [U,d]=solver->SolveLeft_NonSym(E.Flatten(),1e-13,1);
//        cout << std::fixed << std::setprecision(4) << "Left  Arpack n=" << d << endl;
        eigenValue=d(1);
        Left_EigenVector=U.GetColumn(1);
//        cout << "Left_EigenVector=" << Left_EigenVector << endl;
    }
//    VectorCT LEo=Left_EigenVector*Eo.Flatten();
//    cout << "LEo=" << LEo << endl;

//    VectorCT RightEigenVector;
//    {
//        auto [U,d]=solver->SolveLeft_NonSym(Eo.Flatten(),1e-13,1);
//        cout << std::fixed << std::setprecision(4) << "left Arpack o=" << d << endl;
//        eigenValue=d(1);
////        RightEigenVector=U.GetColumn(1);
////        cout << "RightEigenVector=" << RightEigenVector << endl;
//    }

    delete solver;
//    VectorCT EoR=Eo.Flatten()*RightEigenVector;
//    cout << "EoR=" << EoR << endl;
//    cout << "L*EoR=" << Left_EigenVector*EoR << endl;
//    cout << "LEo*R=" << LEo*RightEigenVector << endl;
//    cout << "conj(L)*EoR=" << conj(Left_EigenVector)*EoR << endl;
//    cout << "LEo*conj(R)=" << LEo*conj(RightEigenVector) << endl;
//    cout << "R*EoR=" << RightEigenVector*EoR << endl;
//    cout << "LEo*L=" << LEo*Left_EigenVector << endl;
//    cout << "conj(R)*EoR=" << conj(RightEigenVector)*EoR << endl;
//    cout << "LEo*conj(L)=" << LEo*conj(Left_EigenVector) << endl;

    dcmplx expectation=Left_EigenVector*Eo.Flatten()*conj(Left_EigenVector);
//   cout << "expectation,eigenvalue==" << expectation << " " << eigenValue << endl;
    assert(fabs(imag(expectation))<1e-10);
    return real(expectation);

}

double iTEBDStateImp::GetExpectation (int isite,const MPO* o) const
{
    assert(o->GetL()==2);
    Sites s(isite,const_cast<iTEBDStateImp*>(this));

    int D=s.siteA->GetD1();
    assert(D==s.siteA->GetD2());
    assert(D==s.siteB->GetD1());
    assert(D==s.siteB->GetD2());

    const SiteOperator* soA=o->GetSiteOperator(1);
    const SiteOperator* soB=o->GetSiteOperator(2);
    int DwA1=soA->GetDw12().Dw1;
    int DwA2=soA->GetDw12().Dw2;
    int DwB1=soB->GetDw12().Dw1;
    int DwB2=soB->GetDw12().Dw2;
    assert(DwA1==1);
    assert(DwB2==1);
    assert(DwA2==DwB1);
    Matrix4CT Eo(D,D,D,D);
    Eo.Fill(0);
    for (int na=0; na<itsd; na++)
        for (int nb=0; nb<itsd; nb++)
        {
            MatrixCT theta13_n=s.MA[na]*s.lambdaA*s.MB[nb]*s.lambdaB;
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

                    MatrixCT theta13_m=conj(s.MA[ma])*s.lambdaA*conj(s.MB[mb])*s.lambdaB;
                    assert(theta13_m.GetNumRows()==D);
                    assert(theta13_m.GetNumCols()==D);
                    for (int i1=1; i1<=D; i1++)
                        for (int j1=1; j1<=D; j1++)
                            for (int i3=1; i3<=D; i3++)
                                for (int j3=1; j3<=D; j3++)
                                    Eo(i1,j1,i3,j3)+=theta13_m(i1,i3)*Omn*theta13_n(j1,j3);
                }
        }
    Matrix4CT E=GetTransferMatrix();

    EigenSolver<dcmplx>* solver=new ArpackEigenSolver<dcmplx>;
    VectorCT Left_EigenVector;
    dcmplx eigenValue(0);
    {
        auto [U,d]=solver->SolveLeft_NonSym(E.Flatten(),1e-13,1);
        eigenValue=d(1);
        Left_EigenVector=U.GetColumn(1);
    }
    VectorCT LEo=Left_EigenVector*Eo.Flatten();

    VectorCT RightEigenVector;
    {
        auto [U,d]=solver->SolveRightNonSym(Eo.Flatten(),1e-13,1);
//        cout << std::fixed << std::setprecision(4) << "Right Arpack n=" << d << endl;
        eigenValue=d(1);
        RightEigenVector=U.GetColumn(1);
//        cout << "RightEigenVector=" << RightEigenVector << endl;
    }

    delete solver;
    VectorCT EoR=Eo.Flatten()*RightEigenVector;
//    cout << "EoR=" << EoR << endl;
//    cout << "L*EoR=" << Left_EigenVector*EoR << endl;
//    cout << "LEo*R=" << LEo*RightEigenVector << endl;
//    cout << "conj(L)*EoR=" << conj(Left_EigenVector)*EoR << endl;
//    cout << "LEo*conj(R)=" << LEo*conj(RightEigenVector) << endl;
//    cout << "R*EoR=" << RightEigenVector*EoR << endl;
//    cout << "LEo*L=" << LEo*Left_EigenVector << endl;
//    cout << "conj(R)*EoR=" << conj(RightEigenVector)*EoR << endl;
//    cout << "LEo*conj(L)=" << LEo*conj(Left_EigenVector) << endl;

    dcmplx expectation=Left_EigenVector*Eo.Flatten()*conj(Left_EigenVector);
//    cout << "expectation=" << expectation << " fabs(expectation)=" << fabs(expectation) << endl;
    //assert(fabs(imag(eigenValue))<1e-10);
    return real(expectation);
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


//
//  Same as MPS report except we report one more bond
//
void iTEBDStateImp::Report(std::ostream& os) const
{
    os << "Matrix product state for " << itsL << " lattice sites." << endl;
    os << "  Site  D1  D2  Norm #updates  Emin        Egap     dE" << endl;
    SiteLoop(ia)
    {
        os << std::setw(3) << ia << "  ";
        itsSites[ia]->Report(os);
        os << endl;
    }
    os << "  Bond  D   Rank  Entropy   Min(Sv)   SvError " << endl;
    for (int ib=1; ib<=itsL; ib++)
    {
        os << std::setw(3) << ib << "  ";
        itsBonds[ib]->Report(os);
        os << endl;
    }
}

}
