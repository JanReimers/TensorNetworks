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
    ReCenter(1);
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
        itsBonds.push_back(new Bond(D,epsSV));
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

void iTEBDStateImp::ReCenter(int isite)
{
    s1=Sites(isite,this);
    assert(s1.siteA!=s1.siteB);
    assert(s1.bondA!=s1.bondB);
    assert(s1.siteA->itsLeft_Bond==s1.bondB);
    assert(s1.siteB->itsLeft_Bond==s1.bondA);
    assert(lambdaA().size()>0);
    assert(lambdaB().size()>0);
    assert(Max(lambdaA())>0.0);
    assert(Max(lambdaB())>0.0);
}


iTEBDStateImp::Sites::Sites(int leftSite, iTEBDStateImp* iTEBD)
    : leftSiteNumber(leftSite)
    , siteA(iTEBD->itsSites[iTEBD->GetModSite(leftSite  )])
    , siteB(iTEBD->itsSites[iTEBD->GetModSite(leftSite+1)])
    , bondA(siteA->itsRightBond)
    , bondB(siteB->itsRightBond)
    , GammaA(&siteA->itsMs)
    , GammaB(&siteB->itsMs)
    , lambdaA(&bondA->GetSVs())
    , lambdaB(&bondB->GetSVs())
{

}

iTEBDStateImp::Sites::Sites()
    : leftSiteNumber(1)
    , siteA(nullptr)
    , siteB(nullptr)
    , bondA(nullptr)
    , bondB(nullptr)
//    , GammaA()
//    , GammaB()
//    , lambdaA()
//    , lambdaB()
{

}



void iTEBDStateImp::Normalize(Direction lr)
{
    int D=lambdaA().size();
    Matrix4CT E=GetTransferMatrix(DLeft);
    EigenSolver<dcmplx>* solver=0;
    if (D==1)
        solver=new LapackEigenSolver<dcmplx>;
     else
        solver=new ArpackEigenSolver<dcmplx>;
    dcmplx left_eigenValue(0);
    {
        auto [U,d]=solver->SolveLeft_NonSym(E.Flatten(),1e-13,1);
        left_eigenValue=d(1);
    }
    delete solver;
    assert(fabs(imag(left_eigenValue))<1e-10);
    double lnorm=sqrt(real(left_eigenValue));
//    double rnorm=sqrt(real(right_eigenValue));
//    double fa=itsSites[1]->FrobeniusNorm();
//    double fb=itsSites[2]->FrobeniusNorm();
//    cout << "lnorm, fa,fb=" << lnorm << " " << fa << " " << fb << endl;
    s1.siteA->Rescale(sqrt(lnorm));
    s1.siteB->Rescale(sqrt(lnorm));

}

void iTEBDStateImp::Canonicalize(Direction lr)
{
    ForLoop(lr)
      MPSImp::CanonicalizeSite1(lr,ia,0); //Stores A1-lambda1-A2-lambda2
    ForLoop(lr)
      MPSImp::CanonicalizeSite2(lr,ia,0); //Convert to Gamma1-lambda1-Gamma2-lambda2

}

//void iTEBDStateImp::NormalizeAndCompress(Direction LR,int Dmax,double epsMin);
int iTEBDStateImp::GetModSite(int isite) const
{
    int modSite=((isite-1)%itsL)+1;
    assert(modSite>=1);
    assert(modSite<=itsL);
    return modSite;
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

void iTEBDStateImp::Orthogonalize(SVCompressorC* comp)
{
    //
    //  Build Gamma[n] = GammaA[na]*lambdaA*GammaB[nb]
    //
    dVectorT gamma(itsd*itsd);
    int nab=0;
    for (int nb=0; nb<itsd; nb++)
        for (int na=0; na<itsd; na++,nab++)
            gamma[na+itsd*nb]=GammaA()[na]*lambdaA()*GammaB()[nb];
    //
    //  Run the one site orthogonalization algorithm.
    //
    auto [gammap,lambdap]=Orthogonalize(gamma,lambdaB());
    //
    //  Unpack gammap into GammaA*lambdaA*GammaB, and lambdap into lambdaB.
    //
    UnpackOrthonormal(gammap,lambdap,comp); //No compressions required.
}

void iTEBDStateImp::UnpackOrthonormal(const dVectorT& gammap, DiagonalMatrixRT& lambdap,SVCompressorC* comp)
{
    assert(comp);
    assert(gammap.size()==itsd*itsd);
    int D=lambdaA().size();
    assert(gammap[0].GetLimits()==MatLimits(D,D));
    //
    //  Set lambdaB=lambdap (=lambda prime)
    //
    s1.bondB->SetSingularValues(lambdap,0.0); //Don't use lambdap any more in case it is not normalized
    //
    //  Create lambdaB*gammap*lambdaB and re-shape for SVD
    //
    dVectorT bgb(itsd*itsd);
    for (int n=0; n<itsd*itsd; n++)
        bgb[n]=lambdaB()*gammap[n]*lambdaB(); // Sandwich LambdaB*gammap*LambdaB

    MatrixCT bgb4=ReshapeForSVD(itsd,bgb);
    //
    //  Now unpack  lambdaB*gammap[n]*lambdaB into P[na]*lambdaA'*Q[nb]'
    //
    SVDSolver<dcmplx>* svd_solver=new LapackSVDSolver<dcmplx>();
    auto [P,lambdaA_prime,Q]=svd_solver->SolveAll(bgb4,1e-13); //only keep D svs.
    assert(Max(fabs(P*lambdaA_prime*Q-bgb4))<1e-13);
    delete svd_solver;
//
//  Compress from d*D back to D
//
    double integratedS2 =comp->Compress(P,lambdaA_prime,Q);
    assert(P.GetNumCols()==D);
    assert(P.GetNumRows()==D*itsd);
    assert(Q.GetNumCols()==D*itsd);
    assert(Q.GetNumRows()==D);
//
//  Set and normalize lambdaA.
//
    s1.bondA->SetSingularValues(lambdaA_prime,integratedS2);
//
//  Unpack P into GammaA and Q into GammaB
//
    assert(Min(lambdaB())>1e-10);
    DiagonalMatrixRT lbinv=1.0/lambdaB(); //inverse of LambdaB
    for (int n=0; n<itsd; n++)
        for (int i=1; i<=D; i++)
        for (int j=1; j<=D; j++)
        {
            GammaA()[n](i,j)=lbinv(i)*P(n*D+i,       j);
            GammaB()[n](i,j)=         Q(       i,n*D+j)*lbinv(j);
        }

    TestOrthogonal(Max(D*sqrt(integratedS2),D*D*1e-12));
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

bool iTEBDStateImp::TestOrthogonal(double eps) const
{
    dVectorT gamma(itsd*itsd);
    int D=lambdaA().size();
    int nab=0;
    for (int na=0; na<itsd; na++)
        for (int nb=0; nb<itsd; nb++,nab++)
            gamma[nab]=GammaA()[na]*lambdaA()*GammaB()[nb];
    Matrix4CT Er=GetTransferMatrix(gamma*lambdaB());
    Matrix4CT El=GetTransferMatrix(lambdaB()*gamma);
    MatrixCT I(D,D); //Right ei
    Unit(I);
    MatrixCT ErI=Er*I;
    MatrixCT IEl=I*El;
//    cout << "Er*I=" << ErI << endl;
//    cout << "I*El=" << IEl << endl;
    double r_error= Max(fabs(ErI-I));
    double l_error= Max(fabs(IEl-I));
    if (r_error>D*eps)
    {
        cout << std::scientific;
        cout << "Error not orthogonal Max(fabs(ErI-I))=" << r_error << " > eps=" << eps << endl;
    }
    if (l_error>D*eps)
    {
        cout << std::scientific;
        cout << "Error not orthogonal Max(fabs(IEl-I))=" << l_error << " > eps=" << eps << endl;
    }
    return (r_error<=eps) && (l_error<=eps);
}

iTEBDStateImp::GLType iTEBDStateImp::Orthogonalize(const dVectorT& gamma, const DiagonalMatrixRT& lambda)
{
    int d=gamma.size();
    assert(d>0);
    int D=gamma[0].GetNumRows();
    assert(D==gamma[0].GetNumCols());

//    double s2=lambda.GetDiagonal()*lambda.GetDiagonal();
//    cout << "input lambda s2=" << s2 << endl;

    //
    //  Calculate right and left transfer matrices and the R/L eigen vectors
    //  Transform the eigen vectors into Hermitian eigen matrices
    //
    EigenSolver<dcmplx>* solver=0;
    if (D==1)
        solver=new LapackEigenSolver<dcmplx>;
     else
        solver=new ArpackEigenSolver<dcmplx>;

    dVectorT gl=gamma*lambda;
    for (int n=0;n<d;n++)
        assert(gl[n]==gamma[n]*lambda);
    Matrix4CT Er=GetTransferMatrix(gamma*lambda);

    MatrixCT Vr(D,D); //Right eigen matrix
    double er;
    {
        auto [U,e]=solver->SolveRightNonSym(Er.Flatten(),1e-13,1);
//        cout << "Right e 1-e=" << e << " " << 1.0-e(1) << endl;
//        assert(fabs(1.0-e(1))<1e-10);
        assert(imag(e(1))<1e-10);
        er=real(e(1));
        int ij=1;
        for (int j=1; j<=D; j++)
            for (int i=1; i<=D; i++,ij++)
                Vr(i,j)=U.GetColumn(1)(ij);
        dcmplx phase=Vr(1,1)/fabs(Vr(1,1));
        assert(fabs(phase)-1.0<1e-14);
        Vr*=conj(phase); //Take out arbitrary phase angle
    }
//    if (fabs(er-1.0)>1e-10)
//        cout << "Right eigenvalue=" << er << endl;
//    cout << "Vr=" << Vr << endl;
    double rerr=Max(fabs(Er*Vr-er*Vr));
    if (rerr>1e-13) cout  << std::scientific << "rerr=" << rerr << endl;
    assert(rerr<1e-10);

    Matrix4CT El=GetTransferMatrix(lambda*gamma);
    MatrixCT Vl(D,D); //Right eigen matrix
    double el;
    {
        auto [U,e]=solver->SolveLeft_NonSym(El.Flatten(),1e-13,1);
//        cout << "Left e 1-e=" << e << " " << 1.0-e(1) << endl;
//        assert(fabs(1.0-e(1))<1e-10);
        assert(imag(e(1))<1e-10);
        el=real(e(1));
        int ij=1;
        for (int j=1; j<=D; j++)
            for (int i=1; i<=D; i++,ij++)
                Vl(i,j)=U.GetColumn(1)(ij);
        dcmplx phase=Vl(1,1)/fabs(Vl(1,1));
        assert(fabs(phase)-1.0<1e-14);
        Vl*=conj(phase); //Take out arbitrary phase angle
    }
//    if (fabs(el-1.0)>1e-10)
//        cout << "Left  eigenvalue=" << el << endl;
//    cout << "Vl=" << Vl << endl;
    double lerr=Max(fabs(Vl*El-el*Vl));
    if (lerr>1e-13) cout  << std::scientific << "lerr=" << lerr << endl;
    assert(lerr<1e-10);
    delete solver;
//
//  Decompose eigen matrices
//
    solver=new LapackEigenSolver<dcmplx>; //Switch to a dense solver
    assert(IsHermitian(Vr,1e-10));
    assert(IsHermitian(Vl,1e-10));
    Vr=0.5*(Vr+~Vr);
    Vl=0.5*(Vl+~Vl);
    assert(IsHermitian(Vr,1e-13));
    assert(IsHermitian(Vl,1e-13));
    MatrixCT X,Xinv;
    {
        auto [U,e]=solver->SolveAll(Vr,1e-13);
        X=U*DiagonalMatrix<double>(sqrt(e));
        assert(Min(e)>0.0);
        Xinv=DiagonalMatrix<double>(1.0/sqrt(e))*Transpose(conj(U));
    }
//    cout << "X,Xinv" << X << Xinv << endl;
    assert(IsUnit(X*Xinv,1e-13));
    MatrixCT YT,YTinv;
    {
        auto [U,e]=solver->SolveAll(Vl,1e-13);
        YT=Transpose(U*DiagonalMatrix<double>(sqrt(e)));
        assert(Min(e)>0.0);
        YTinv=Transpose(DiagonalMatrix<double>(1.0/sqrt(e))*Transpose(conj(U)));
    }
//    cout << "YT*YTinv" << YT*YTinv << endl;
    assert(IsUnit(YT*YTinv,1e-12));
    delete solver;
    //
    //  Transform lambda
    //
    MatrixCT YlX=YT*lambda*X; //THis will scale lambda by 1/D since YT and X are both O(1/sqrt(D))
    SVDSolver<dcmplx>* svd_solver=new LapackSVDSolver<dcmplx>();
    auto [U,lambda_prime,VT]=svd_solver->SolveAll(YlX,1e-13);
    delete svd_solver;

//    cout << U << lambda_prime << VT << endl;
    //
    //  Transform Gamma
    //
    dVectorT gamma_prime(d);
    for (int n=0;n<d;n++)
    {
        gamma_prime[n]=VT*Xinv*gamma[n]*YTinv*U;
//        cout << gamma_prime[n] << endl;
    }
    //
    //  Verify orthogonaly
    //
    Er=GetTransferMatrix(gamma_prime*lambda_prime);
    El=GetTransferMatrix(lambda_prime*gamma_prime);
    MatrixCT I(D,D); //Right ei
    Unit(I);
    MatrixCT ErI=Er*I-er*I;
    MatrixCT IEl=I*El-el*I;
//    cout << "Max(fabs(Er*I-er*I))=" << Max(fabs(Er*I-er*I))  << endl;
//    cout << "Max(fabs(I*El-el*I))=" << Max(fabs(I*El-el*I))  << endl;
    assert(Max(fabs(ErI))<D*D*1e-6);
    assert(Max(fabs(IEl))<D*D*1e-6);
//    cout << std::scientific << "Er*I=" << ErI-I << endl;
//    cout << "i*El=" << IEl-I << endl;
//    assert(IsUnit(ErI,D*1e-9));
//    assert(IsUnit(IEl,D*1e-9));

    return std::make_tuple(gamma_prime,lambda_prime);
}

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


//    Matrix4CT E(D,D,D,D);
//    E.Fill(0);
//    for (int na=0; na<itsd; na++)
//        for (int nb=0; nb<itsd; nb++)
//        {
//            MatrixCT theta13=GammaA()[na]*lambdaA()*GammaB()[nb]*lambdaB();
//            MatrixCT theta13c=conj(GammaA()[na])*lambdaA()*conj(GammaB()[nb])*lambdaB();
//            assert(conj(theta13c)==theta13);
//            assert(theta13.GetNumRows()==D);
//            assert(theta13.GetNumCols()==D);
//            for (int i1=1; i1<=D; i1++)
//                for (int j1=1; j1<=D; j1++)
//                    for (int i3=1; i3<=D; i3++)
//                        for (int j3=1; j3<=D; j3++)
//                            E(i1,j1,i3,j3)+=theta13c(i1,i3)*theta13(j1,j3);
//        }
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

iTEBDStateImp::MdType iTEBDStateImp::GetEigenMatrix(TensorNetworks::Direction lr, const Matrix4CT& theta)
{
    int D=s1.siteA->GetD1();
    EigenSolver<dcmplx>* solver=0;
    if (D==1)
        solver=new LapackEigenSolver<dcmplx>;
     else
        solver=new ArpackEigenSolver<dcmplx>;

    VectorCT eigenVector;
    double   eigenValue;
    switch (lr)
    {
        case DLeft :
         {
            auto [U,e]=solver->SolveLeft_NonSym(theta.Flatten(),1e-13,1);
            assert(imag(e(1))<1e-13);
            eigenValue=real(e(1));
            eigenVector=U.GetColumn(1);
            break;
        }
       case DRight:
        {
            auto [U,e]=solver->SolveRightNonSym(theta.Flatten(),1e-13,1);
            assert(imag(e(1))<1e-13);
            eigenValue=real(e(1));
            eigenVector=U.GetColumn(1);
            break;
        }
    }
    cout << "eigen value=" << eigenValue << endl;
    //
    //  Unpack eigenVector into a matrix
    //
    assert(eigenVector.size()==D*D);
    MatrixCT V(D,D); // eigen matrix
    int ij=1;
    for (int j=1; j<=D; j++)
        for (int i=1; i<=D; i++,ij++)
            V(i,j)=eigenVector(ij);

    dcmplx phase=V(1,1)/fabs(V(1,1));
    assert(fabs(phase)-1.0<1e-14);
    V*=conj(phase); //Take out arbitrary phase angle

    double err;
    switch (lr)
    {
        case DLeft  : err=Max(fabs(V*theta-eigenValue*V));break; //Assumes eigen value was 1
        case DRight : err=Max(fabs(theta*V-eigenValue*V));break;
    }
    if (err>1e-13) cout  << std::scientific << "Eigen vector error=" << err << endl;
    assert(err<1e-10);
    return std::make_tuple(V,eigenValue);
}

//
//  Decompose eigen matrices
//
iTEBDStateImp::MMType iTEBDStateImp::Factor(const MatrixCT m)
{
    EigenSolver<dcmplx>* solver=new LapackEigenSolver<dcmplx>; //Switch to a dense solver
    assert(IsHermitian(m,1e-10));  //Make sure are close to Hermitian
    MatrixCT mh=0.5*(m+~m); //Make it perfectly hermitian.  THis should cancel out some numerical round off noise.
    assert(IsHermitian(mh,1e-13));
    MatrixCT X,Xinv;
    auto [U,e]=solver->SolveAll(mh,1e-13);
    delete solver;
    X=U*DiagonalMatrix<double>(sqrt(e));
    assert(Min(e)>1e-10);
    Xinv=DiagonalMatrix<double>(1.0/sqrt(e))*Transpose(conj(U));
    assert(IsUnit(X*Xinv,1e-13));
    return std::make_tuple(X,Xinv);
}

//-----------------------------------------------------------------------------
//
//  THis follows PHYSICAL REVIEW B 78, 155117 2008 figure 14
//
void iTEBDStateImp::Apply(const Matrix4RT& expH,SVCompressorC* comp)
{
    assert(comp);
//    assert(TestOrthogonal(1e-9));
    //
    //  Make sure everything is square
    assert(s1.siteA->GetD2()==s1.siteB->GetD1());
    assert(s1.siteA->GetD1()==s1.siteB->GetD2());
    assert(s1.siteA->GetD1()==s1.siteA->GetD2());
    int D=s1.siteA->GetD1();

    dVectorT  Thetap(itsd*itsd);
    for (int n=0;n<itsd*itsd;n++)
    {
        Thetap[n].SetLimits(D,D);
        Fill(Thetap[n],dcmplx(0.0));
    }

    for (int mb=0; mb<itsd; mb++)
    for (int ma=0; ma<itsd; ma++)
    {
        MatrixCT theta13  =GammaA()[ma]*lambdaA()*GammaB()[mb];  //Figure 14 v
        int nab=0;
        for (int nb=0; nb<itsd; nb++)
        for (int na=0; na<itsd; na++,nab++)
            Thetap[nab]+= theta13*expH(ma,na,mb,nb);
    }

    auto [gammap,lambdap]=Orthogonalize(Thetap,lambdaB());
    UnpackOrthonormal(gammap,lambdap,comp);
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
    assert(fabs(imag(expectation1))<1e-10);
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
    assert(fabs(imag(expectation1))<1e-10);
    return real(expectation1);

}

double iTEBDStateImp::GetExpectation (const MPO* o) const
{
    assert(o->GetL()==2);
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

    assert(fabs(imag(expectation1))<1e-10);
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
