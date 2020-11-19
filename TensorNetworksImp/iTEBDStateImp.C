#include "TensorNetworksImp/iTEBDStateImp.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/MPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworks/Factory.H"

#include "TensorNetworksImp/Bond.H"

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

void iTEBDStateImp::IncreaseBondDimensions(int D)
{
    for (int ia=1; ia<=itsL; ia++)
    {
        itsSites[ia]->NewBondDimensions(D,D,true);
        itsBonds[ia]->NewBondDimension(D);
    }
}


void iTEBDStateImp::ReCenter(int isite) const
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


iTEBDStateImp::Sites::Sites(int leftSite, const iTEBDStateImp* iTEBD)
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


double iTEBDStateImp::FindiTimeGroundState(const Hamiltonian* H,const IterationSchedule& is)
{
    assert(Logger); //Make sure we have global logger.
    Canonicalize(TensorNetworks::DLeft);
    Logger->LogInfoV(1,"Initiate iTime GS iterations, Dmax=%4d, Norm status=%s",GetMaxD(),GetNormStatus().c_str());

    double E1=0;
    for (is.begin(); !is.end(); is++)
        E1=FindiTimeGroundState(H,*is);

    MPO* H2=H->CreateH2Operator();
    double E2=GetExpectation(H2);
    E2= E2-E1*E1;
    delete H2;
    Logger->LogInfoV(0,"Finished iTime GS iterations D=%4d, <E>=%.9f, <E^2>-<E>^2=%.2e",GetMaxD(),E1,E2);
    return E2;
}

double iTEBDStateImp::FindiTimeGroundState(const Hamiltonian* H,const IterationScheduleLine& isl)
{
    assert(isl.itsDmax>0 || isl.itsEps.itsMPSCompressEpsilon>0);


    Matrix4RT Hlocal=H->BuildLocalMatrix();
    Matrix4RT expH=Hamiltonian::ExponentH(isl.itsdt,Hlocal);
    int Dmax=GetMaxD();
    Logger->LogInfoV(1,"   Begin iterations, dt=%.3f,  GetMaxD=%4d, isl.Dmax=%4d",isl.itsdt,Dmax,isl.itsDmax);
    SVCompressorC* mps_compressor =Factory::GetFactory()->MakeMPSCompressor(Dmax,0.0);
    Orthogonalize(mps_compressor);
    ReCenter(1);
    double E1=GetExpectation(H)/(itsL-1);

    for (int D=Dmax;D<=isl.itsDmax;D+=isl.itsDeltaD)
    {
        if (D>Dmax)
        {
            IncreaseBondDimensions(D);
            delete mps_compressor;
            mps_compressor =Factory::GetFactory()->MakeMPSCompressor(D,0.0);
        }
        int niter=1;
        for (; niter<isl.itsMaxGSSweepIterations; niter++)
        {
            Apply(expH,mps_compressor);
            ReCenter(2);
            Apply(expH,mps_compressor);
            ReCenter(1);
            double Enew=GetExpectation(H)/(itsL-1);
            double dE=Enew-E1;
            E1=Enew;
            Logger->LogInfoV(2,"      E=%.9f, dE=%.2e, niter=%4d",E1,dE,niter);
            if (fabs(dE)<=isl.itsEps.itsDelatEnergy1Epsilon || dE>0.0) break;
        }
        Orthogonalize(mps_compressor);
        E1=GetExpectation(H)/(itsL-1);
        Logger->LogInfoV(2,"      E=%.9f, niter=%4d",E1,niter);
        Logger->LogInfoV(1,"   End  iterations, dt=%.3f,   E=%.9f, D=%4d, isl.Dmax=%4d",isl.itsdt,E1,D,isl.itsDmax);
    }
    Logger->LogInfoV(1,"   End D iterations, dt=%.3f, D=%4d, E=%.9f.",isl.itsdt,GetMaxD(),E1);
    delete mps_compressor;
    return E1;
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

ONErrors iTEBDStateImp::Orthogonalize(SVCompressorC* comp)
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
    DiagonalMatrixRT lb=lambdaB();
    auto [gammap,lambdap]=Orthogonalize(gamma,lb);
    //
    //  Unpack gammap into GammaA*lambdaA*GammaB, and lambdap into lambdaB.
    //
    return UnpackOrthonormal(gammap,lambdap,comp); //No compressions required.
}

ONErrors iTEBDStateImp::UnpackOrthonormal(const dVectorT& gammap, DiagonalMatrixRT& lambdap,SVCompressorC* comp)
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
    double compessionError =comp->Compress(P,lambdaA_prime,Q);
    assert(P.GetNumCols()==D);
    assert(P.GetNumRows()==D*itsd);
    assert(Q.GetNumCols()==D*itsd);
    assert(Q.GetNumRows()==D);
//
//  Set and normalize lambdaA.
//
    s1.bondA->SetSingularValues(lambdaA_prime,compessionError);
//
//  Unpack P into GammaA and Q into GammaB
//
    if (Min(lambdaB())<1e-10)
        std::cerr << "Warning  small lambda min(lambda)=" << Min(lambdaB()) << std::endl;
    DiagonalMatrixRT lbinv=1.0/lambdaB(); //inverse of LambdaB
    for (int n=0; n<itsd; n++)
        for (int i=1; i<=D; i++)
        for (int j=1; j<=D; j++)
        {
            GammaA()[n](i,j)=lbinv(i)*P(n*D+i,       j);
            GammaB()[n](i,j)=         Q(       i,n*D+j)*lbinv(j);
        }

    return GetOrthonormalityErrors();
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

ONErrors iTEBDStateImp::GetOrthonormalityErrors() const
{
    dVectorT gamma(itsd*itsd);
    int D=lambdaA().size();
    MatrixCT I(D,D); //Right ei
    Unit(I);

    int nab=0;
    for (int na=0; na<itsd; na++)
        for (int nb=0; nb<itsd; nb++,nab++)
            gamma[nab]=GammaA()[na]*lambdaA()*GammaB()[nb];

    MatrixCT Nr=GetNormMatrix(DRight,gamma*lambdaB());
    MatrixCT Nl=GetNormMatrix(DLeft ,lambdaB()*gamma);
    dcmplx right_norm=Sum(Nr.GetDiagonal())/static_cast<double>(D);
    dcmplx left__norm=Sum(Nl.GetDiagonal())/static_cast<double>(D);
    Nr/=right_norm; //Get diagonals as close 1.0 as we can
    Nl/=left__norm;

    double right_norm_error=fabs(right_norm-1.0);
    double left__norm_error=fabs(left__norm-1.0);
    double right_orth_error= FrobeniusNorm(Nr-I);
    double left__orth_error= FrobeniusNorm(Nl-I);

    return {right_norm_error,left__norm_error,right_orth_error,left__orth_error};
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
            if (fabs(imag(e(1)))>1e-13)
                std::cerr << std::scientific << "Warning: Dominant eigenvalue has large imaginary component e=" << e(1) << std::endl;
//            assert(imag(e(1))<1e-13);
            eigenValue=real(e(1));
            eigenVector=U.GetColumn(1);
            break;
        }
       case DRight:
        {
            auto [U,e]=solver->SolveRightNonSym(theta.Flatten(),1e-13,1);
            if (fabs(imag(e(1)))>1e-13)
                std::cerr << std::scientific << "Warning: Dominant eigenvalue has large imaginary component e=" << e(1) << std::endl;
//            assert(imag(e(1))<1e-13);
            eigenValue=real(e(1));
            eigenVector=U.GetColumn(1);
            break;
        }
    }
//    cout << "eigen value=" << eigenValue << endl;
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
        case DLeft  : err=Max(fabs(V*theta-eigenValue*V));break;
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
    if (Min(e)<1e-10)
    {
        std::cerr << "iTEBDStateImp::Factor Warning  small eigen min(e)=" << Min(e) << std::endl;
        std::cerr << "  mh.diag=" << mh.GetDiagonal() << std::endl;
        std::cerr << "   e=" << e << std::endl;
    }

    Xinv=DiagonalMatrix<double>(1.0/sqrt(e))*Transpose(conj(U));
    assert(IsUnit(X*Xinv,1e-13));
    return std::make_tuple(X,Xinv);
}

//
// Iterative version Ho N. Phien, Ian P. McCulloch, and Guifré Vidal, "Fast convergence of imaginary
// time evolution tensor network algorithms by recycling the environment", Physical Review B 91, 11 (2015).
//
iTEBDStateImp::GLType iTEBDStateImp::OrthogonalizeI(dVectorT& gamma, DiagonalMatrixRT& lambda)
{
    int d=gamma.size();
    assert(d>0);
    int D=gamma[0].GetNumRows();
    assert(D==gamma[0].GetNumCols());
    double eps=1e-6;
    MatrixCT Vr,Vl,I(D,D); //Right ei
    Unit(I);
    SVDSolver<dcmplx>* svd_solver=new LapackSVDSolver<dcmplx>();
    dcmplx er;
    dcmplx el;
    int niter=0;
    do
    {
        Vr=GetNormMatrix(DRight,gamma*lambda); //=Er*I
        Vl=GetNormMatrix(DLeft ,lambda*gamma); //=I*El
        if (Min(fabs(Vr.GetDiagonal()))<1e-10)
        {
            cout << "Singular Vr, bailing" << endl;
            return std::make_tuple(gamma,lambda); //Bail if V is singular. THis can happen when increasing D.
        }
        if (Min(fabs(Vl.GetDiagonal()))<1e-10)
        {
            cout << "Singular Vl, bailing" << endl;
            return std::make_tuple(gamma,lambda); //Bail if V is singular. THis can happen when increasing D.
        }
//
//  Try to normalize
//
        er=Vr(1,1);
        el=Vl(1,1);
        Vr/=er;
        Vl/=el;

//
//  Make sure Vr and Vl are loosely Hermitian.
//
        assert(IsHermitian(Vr,1e-10));
        assert(IsHermitian(Vl,1e-10));
    //
    //  Decompose eigen matrices
    //
        auto [X ,Xinv ]=Factor(Vr);
        auto [Yd,Ydinv]=Factor(Vl);
//        MatrixCT YT   =conj(Yd); Y_Transpose should be conj(Ydagger) but this doesn't seem to work!!
//        MatrixCT YTinv=conj(Ydinv);
        MatrixCT YT   =Transpose(Yd);
        MatrixCT YTinv=Transpose(Ydinv);
        assert(IsUnit(X*Xinv,1e-13));
        assert(IsUnit(YT*YTinv,1e-12));
        //
        //  Transform lambda and SVD
        //
        MatrixCT YlX=YT*lambda*X; //THis will scale lambda by 1/D since YT and X are both O(1/sqrt(D))
        auto [U,lambda_prime,VT]=svd_solver->SolveAll(YlX,1e-13);
        //
        //  Normalize lambda'
        //
        double s2=lambda_prime.GetDiagonal()*lambda_prime.GetDiagonal();
        lambda_prime/=sqrt(s2);
        //
        //  Transform Gamma
        //
        for (int n=0;n<d;n++)
        {
            MatrixCT gamma_prime=VT*Xinv*gamma[n]*YTinv*U;
            gamma[n]=gamma_prime;
        }
        double deltal=Max(fabs(lambda-lambda_prime));
        lambda=lambda_prime;
//        cout << "deltal,er,el,er-el=" << deltal << " " << fabs(er-1.0) << " " << fabs(el-1.0) << " " << fabs(er-el) << endl;
        if (deltal<eps) break;
        niter++;
    } while (niter<100);
    delete svd_solver;
//
//  Normalize
//
        for (int n=0;n<d;n++)
            gamma[n]/=sqrt(er);

    //
    //  Verify orthogonaly
    //
    MatrixCT Nr=GetNormMatrix(DRight,gamma*lambda); //=Er*I
    MatrixCT Nl=GetNormMatrix(DLeft ,lambda*gamma); //=I*El
    double  left_error=Max(fabs(Nr-I));
    double right_error=Max(fabs(Nl-I));
    if (left_error>D*eps)
    {
        cout << std::scientific << "Warning: Left orthogonality error=" << left_error  << endl;
        cout << "Nl=" << Nl << endl;
    }
    if (right_error>D*eps)
    {
        cout << std::scientific << "Warning: Right orthogonality error=" << right_error  << endl;
        cout << "Nr=" << Nr << endl;
    }
//    cout << niter << " " << D << endl;

    return std::make_tuple(gamma,lambda);
}

//
// Eigen vector version as per PHYSICAL REVIEW B 78, 155117 2008
//
iTEBDStateImp::GLType iTEBDStateImp::Orthogonalize(dVectorT& gamma, const DiagonalMatrixRT& lambda)
{
    int d=gamma.size();
    assert(d>0);
    int D=gamma[0].GetNumRows();
    assert(D==gamma[0].GetNumCols());
    MatrixCT I(D,D); //Right ei
    Unit(I);
    //
    //  Calculate right and left transfer matrices and the R/L eigen vectors
    //  Transform the eigen vectors into Hermitian eigen matrices
    //
    Matrix4CT Er=GetTransferMatrix(gamma*lambda);
    Matrix4CT El=GetTransferMatrix(lambda*gamma);
//    cout << "Er*I=" << Er*I << endl;
//    cout << "I*El=" << I*El << endl;

    auto [Vr,er]=GetEigenMatrix(DRight,Er);
    if (Min(fabs(Vr.GetDiagonal()))<1e-10) return std::make_tuple(gamma,lambda); //Bail if V is singular. THis can happen when increasing D.

    auto [Vl,el]=GetEigenMatrix(DLeft ,El);
    if (Min(fabs(Vl.GetDiagonal()))<1e-10) return std::make_tuple(gamma,lambda); //Bail if V is singular. THis can happen when increasing D.
//
//  Normalize
//
    if (fabs(er-el)>=2e-13)
        cout << fabs(er-el) << endl;
    assert(fabs(er-el)<2e-13);
    Er.Flatten()/=er;
    El.Flatten()/=el;
    s1.siteA->Rescale(sqrt(sqrt(er)));
    s1.siteB->Rescale(sqrt(sqrt(er)));
    for (int n=0;n<d;n++)
        gamma[n]/=sqrt(er);
//
// Check eigen matrix accuracy.
//
    double rerr=Max(fabs(Er*Vr-Vr));
    if (rerr>1e-13) cout  << std::scientific << "rerr=" << rerr << endl;
    assert(rerr<1e-10);

    double lerr=Max(fabs(Vl*El-Vl));
    if (lerr>1e-13) cout  << std::scientific << "lerr=" << lerr << endl;
    assert(lerr<1e-10);
//
//  Make sure Vr and Vl are as Hermitian as they can get.
//
    assert(IsHermitian(Vr,1e-10));
    assert(IsHermitian(Vl,1e-10));
    Vr=0.5*(Vr+~Vr); //Try and clean up non-Hermitian round-off noise.
    Vl=0.5*(Vl+~Vl);
    assert(IsHermitian(Vr,1e-13));
    assert(IsHermitian(Vl,1e-13));

//
//  Decompose eigen matrices
//
    auto [X,Xinv]=Factor(Vr);
    auto [Y,Yinv]=Factor(Vl);
    MatrixCT YT   =Transpose(Y);
    MatrixCT YTinv=Transpose(Yinv);
    assert(IsUnit(X*Xinv,1e-13));
    //assert(IsUnit(YT*YTinv,1e-12));
    //
    //  Transform lambda and SVD
    //
    MatrixCT YlX=YT*lambda*X; //THis will scale lambda by 1/D since YT and X are both O(1/sqrt(D))
    SVDSolver<dcmplx>* svd_solver=new LapackSVDSolver<dcmplx>();
    auto [U,lambda_prime,VT]=svd_solver->SolveAll(YlX,1e-13);
    delete svd_solver;
    //
    //  Transform Gamma
    //
    dVectorT gamma_prime(d);
    for (int n=0;n<d;n++)
    {
        gamma_prime[n]=VT*Xinv*gamma[n]*YTinv*U;
    }
    //
    //  Verify orthogonaly
    //
    MatrixCT Nr=GetNormMatrix(DRight,gamma_prime*lambda_prime); //=Er*I
    MatrixCT Nl=GetNormMatrix(DLeft ,lambda_prime*gamma_prime); //=I*El
    double  left_error=Max(fabs(Nr-I));
    double right_error=Max(fabs(Nl-I));
    if (left_error>1e-12)
    {
        cout << std::scientific << "Warning: Left orthogonality error=" << left_error  << endl;
    }
    if (right_error>1e-12)
    {
        cout << std::scientific << "Warning: Right orthogonality error=" << right_error  << endl;
    }

    return std::make_tuple(gamma_prime,lambda_prime);
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

    DiagonalMatrixRT lb=lambdaB();
//    auto [gammap,lambdap]  =Orthogonalize(Thetap,lb);
//    auto [gammapI,lambdapI]=OrthogonalizeI(gammap,lambdap);
    auto [gammap,lambdap]=OrthogonalizeI(Thetap,lb);
//    cout << "Max(fabs(lambdapI-lambdap))=" << Max(fabs(lambdapI.GetDiagonal()-lambdap.GetDiagonal())) << endl;
//    cout << "lambdapI=" << lambdapI.GetDiagonal() << endl;
//    cout << "lambdap =" << lambdap .GetDiagonal()  << endl;
//    assert(false);
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
        std::cerr << "Warning high imaginary part in expectation " << expectation1 << std::endl;
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
