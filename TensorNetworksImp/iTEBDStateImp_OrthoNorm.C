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
#include "NumericalMethods/LapackSVDSolver.H"
#include "oml/cnumeric.h"
#include <iomanip>

namespace TensorNetworks
{



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
    Logger->LogInfoV(2,"iTEBDStateImp::Normalize eigenvalue=(%.5f,%.1e)",real(left_eigenValue),imag(left_eigenValue));
    if (fabs(imag(left_eigenValue))>1e-10)
        Logger->LogWarnV(0,"iTEBDStateImp::Normalize eigenvalue=(%.5f,%.1e) has large imaginary component",real(left_eigenValue),imag(left_eigenValue));

    double lnorm=sqrt(real(left_eigenValue));
    s1.siteA->Rescale(sqrt(lnorm));
    s1.siteB->Rescale(sqrt(lnorm));

}

double iTEBDStateImp::GetOrthonormalityErrors() const
{
    return GetOrthonormalityErrors(ContractAlB(),lambdaB());
}

double iTEBDStateImp::GetOrthonormalityErrors(const dVectorT& gamma, const DiagonalMatrixRT& lambda)
{
    int D=lambda.size();
    MatrixCT I(D,D); //Right ei
    Unit(I);

    MatrixCT Nr=GetNormMatrix(DRight,gamma*lambda);
    MatrixCT Nl=GetNormMatrix(DLeft ,lambda*gamma);
    dcmplx right_norm=Sum(Nr.GetDiagonal())/static_cast<double>(D);
    dcmplx left__norm=Sum(Nl.GetDiagonal())/static_cast<double>(D);
    Nr/=right_norm; //Get diagonals as close 1.0 as we can
    Nl/=left__norm;

    double right_norm_error=fabs(right_norm-1.0);
    double left__norm_error=fabs(left__norm-1.0);
    double right_orth_error= FrobeniusNorm(Nr-I);
    double left__orth_error= FrobeniusNorm(Nl-I);
    if (right_norm_error>D*right_orth_error && right_norm_error>1e-13)
        Logger->LogWarnV(0,"iTEBDStateImp::GetOrthonormalityErrors large right norm error=%.1e, orth error=%.1e", right_norm_error,right_orth_error );
    if (left__norm_error>D*left__orth_error && left__norm_error>1e-13)
        Logger->LogWarnV(0,"iTEBDStateImp::GetOrthonormalityErrors large left  norm error=%.1e, orth error=%.1e", left__norm_error,right_orth_error );


    return Max(right_orth_error,left__orth_error);
}


double iTEBDStateImp::OrthogonalizeI(SVCompressorC* comp, double eps, int niter)
{
    assert(eps>0.0);
    //
    //  Build Gamma[n] = GammaA[na]*lambdaA*GammaB[nb]
    //
    dVectorT gamma=ContractAlB();
    //
    //  Run the one site orthogonalization algorithm.
    //
    DiagonalMatrixRT lambda=lambdaB();
    OrthogonalizeI(gamma,lambda,eps,niter);
    //
    //  Unpack gammap into GammaA*lambdaA*GammaB, and lambdap into lambdaB.
    //
    s1.bondB->SetSingularValues(lambda,0.0);
    return UnpackOrthonormal(gamma,comp); //No compressions required.
}


double iTEBDStateImp::Orthogonalize(SVCompressorC* comp)
{
    //
    //  Build Gamma[n] = GammaA[na]*lambdaA*GammaB[nb]
    //
    dVectorT gamma=ContractAlB();
    //
    //  Run the one site orthogonalization algorithm.
    //
    DiagonalMatrixRT lambda=lambdaB();
    Orthogonalize(gamma,lambda);
    //
    //  Unpack gammap into GammaA*lambdaA*GammaB, and lambdap into lambdaB.
    //
    s1.bondB->SetSingularValues(lambda,0.0);
    return UnpackOrthonormal(gamma,comp); //No compressions required.
}

double iTEBDStateImp::UnpackOrthonormal(const dVectorT& gammap, SVCompressorC* comp)
{
    assert(comp);
    assert(gammap.size()==itsd*itsd);
    int D=lambdaA().size();
    assert(gammap[0].GetLimits()==MatLimits(D,D));
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
//    cout << "lambdaA_prime=" << lambdaA_prime.GetDiagonal() << endl;
    double compessionError =comp->Compress(P,lambdaA_prime,Q);
//    cout << "lambdaA_prime=" << lambdaA_prime.GetDiagonal() << " " << D << " " << itsd << endl;
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
    if (Min(lambdaB())<1e-13)
        Logger->LogWarnV(0,"iTEBDStateImp::UnpackOrthonormal small lambda min(lambda)=%.1e", Min(lambdaB()) );

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


//
//  SVD gamma into GammaA, lambdaA, GammaB
//
void iTEBDStateImp::Unpack(const dVectorT& gamma,SVCompressorC* comp)
{
//    int d=gamma.size();
    int D2=lambdaA().size();
    int D1=lambdaB().size();
//    cout << "d,D1,D2,gamma[0]=" << d << " " << D1 << " " << D2 << " " << gamma[0].GetLimits() << endl;
//    assert(gamma[0].GetLimits()==MatLimits(D1,D2));

    MatrixCT bgb4=ReshapeForSVD(itsd,gamma);
//    cout << "bgb4=" << bgb4.GetLimits() << endl;
    //
    //  Now unpack  lambdaB*gammap[n]*lambdaB into P[na]*lambdaA'*Q[nb]'
    //
    SVDSolver<dcmplx>* svd_solver=new LapackSVDSolver<dcmplx>();
    auto [U,s,Vd]=svd_solver->SolveAll(bgb4,1e-13); //only keep D svs.
    double svdError=Max(fabs(U*s*Vd-bgb4));
    if (svdError>bgb4.GetNumRows()*bgb4.GetNumCols()*1e-13)
    {
//        Logger->LogWarnV(0,"iTEBDStateImp::Unpack large SVD error=%.1e",svdError  );
    }
    delete svd_solver;
    double trunc_error=0.0;
//    cout << "U=" << U.GetLimits() << endl;
//    cout << "s=" << s.GetLimits() << endl;
//    cout << "Vd=" << Vd.GetLimits() << endl;
//    cout << "s=" << s.GetDiagonal() << endl;
    if (comp) trunc_error=comp->Compress(U,s,Vd);
//    cout << "U=" << U.GetLimits() << endl;
//    cout << "s=" << s.GetLimits() << endl;
//    cout << "Vd=" << Vd.GetLimits() << endl;
//
//  Set and normalize lambdaA.
//
    s1.bondA->SetSingularValues(s,trunc_error);
//
//  Unpack P into GammaA and Q into GammaB
//
    if (Min(lambdaA())<1e-13)
    {
        // Did we forget to remove smalle SVs on this call??
        Logger->LogWarnV(0,"iTEBDStateImp::Unpack small lambdaA min(lambda)=%.1e", Min(lambdaA()) );
        cout << "LambdaA=" << lambdaA().GetDiagonal() << endl;
    }

    if (Min(lambdaB())<1e-13)
    {
        // Did we forget to remove smalle SVs on the previous call??
        Logger->LogWarnV(0,"iTEBDStateImp::Unpack small lambdaB min(lambda)=%.1e", Min(lambdaB()) );
        cout << "LambdaB=" << lambdaB().GetDiagonal() << endl;
    }

    D1=U.GetNumRows()/itsd;
    D2=s.GetDiagonal().size();
    NewBondDimensions(D1,D2);
    DiagonalMatrixRT lbinv=1.0/lambdaB(); //inverse of LambdaB
    for (int n=0; n<itsd; n++)
        for (int i=1; i<=D1; i++)
        for (int j=1; j<=D2; j++)
        {
            GammaA()[n](i,j)=lbinv(i)*U (n*D1+i,     j);
            GammaB()[n](j,i)=         Vd(     j,n*D1+i)*lbinv(i);
        }

}
void iTEBDStateImp::CalculateLambdaB(const dVectorT& theta_BA,SVCompressorC* comp)
{
    assert(comp);
    MatrixCT theta4=ReshapeForSVD(itsd,theta_BA);
    SVDSolver<dcmplx>* svd_solver=new LapackSVDSolver<dcmplx>();
    auto [U,s,Vd]=svd_solver->SolveAll(theta4,1e-13); //only keep D svs.
    double svdError=Max(fabs(U*s*Vd-theta4));
    if (svdError>theta4.GetNumRows()*theta4.GetNumCols()*1e-13)
    {
        Logger->LogWarnV(0,"iTEBDStateImp::CalculateLambdaB large SVD error=%.1e",svdError  );
    }
    double trunct_error=0.0;//comp->Compress(U,s,Vd);
    s1.bondB->SetSingularValues(s,trunct_error); //Keep all SVs for now
//    cout << "LambdaB=" << lambdaB().GetDiagonal() << endl;
}


// move physical indices from inside to outside.
//
//    U(i,n)(j) -> U(i)(j,n)
//    V(i)(j,n) -> V(i,n)(j)
//
std::tuple<MatrixCT,MatrixCT> ReshapeVU(int d,const MatrixCT& Vd,const MatrixCT& U)
{
    int Dx=U.GetNumCols();
    int DDw=U.GetNumRows()/d;
    assert( U.GetNumRows()==d*DDw);
    assert(Vd.GetNumCols()==d*DDw);
    assert(Vd.GetNumRows()==  Dx);
    MatrixCT Ur(DDw,d*Dx),Vdr(d*Dx,DDw);
    for (int n=0;n<d;n++)
    {
        int nDx =n*Dx;
        int nDDw=n*DDw;
        for (int i=1;i<=DDw;i++)
            for (int j=1;j<=Dx;j++)
            {
                Ur (i    ,j+nDx)=U (i+nDDw,j     );
                Vdr(j+nDx,i    )=Vd(j     ,i+nDDw);
            }
    }
    return std::make_tuple(Vdr,Ur);
}

DiagonalMatrixRT Extend(const DiagonalMatrixRT& lambda,int Dw)
{
    //
    //  Now load Dw copies of lb into the extended lambdaB
    //
    int D=lambda.GetDiagonal().size();
    DiagonalMatrixRT extended_lambda(D*Dw);
    int iw=1;
    for (int w=1;w<=Dw;w++)
        for (int i=1;i<=D;i++,iw++)
            extended_lambda(iw)=lambda(i,i);

    return extended_lambda;
}
//
//  See section on notes "Contracting iTEBD state with an iMPO"
//
void iTEBDStateImp::Unpack_iMPO(const dVectorT& theta_BA,SVCompressorC* D_compressor)
{
    assert(D_compressor);
    int D=lambdaA().GetDiagonal().size();
    assert(D==lambdaB().GetDiagonal().size());
    MatrixCT theta_BAr=ReshapeForSVD(itsd,theta_BA);
//    cout << "theta_BAr=" << theta_BAr.GetLimits() << endl;
    assert(theta_BA[0].GetNumRows()%D==0);
    int Dw=theta_BA[0].GetNumRows()/D;
    //
    //  Now SVD theta_BAr to for U*lambdaA*Vd
    //
    SVDSolver<dcmplx>* svd_solver=new LapackSVDSolver<dcmplx>();
    MatrixCT theta_AB;
    {
        auto [U,s,Vd]=svd_solver->SolveAll(theta_BAr,1e-13);
        double svdError=Max(fabs(U*s*Vd-theta_BAr));
        if (svdError>theta_BAr.GetNumRows()*theta_BAr.GetNumCols()*1e-13)
        {
            Logger->LogWarnV(0,"iTEBDStateImp::Unpack_iMPO large #1 SVD error=%.1e",svdError  );
        }
//        cout << "SVD #1 Min(s)=" << Min(s) << endl;
        SVCompressorC* eps_compressor =Factory::GetFactory()->MakeMPSCompressor(0.0,1e-13);
        double trunc_error=D_compressor->Compress(U,s,Vd); //Compress down to D
        delete eps_compressor;
        int Dx=s.GetDiagonal().size();
//        cout << "Dx=" << Dx << endl;
    //    cout << s.GetDiagonal() << endl;
        s1.bondB->SetSingularValues(s,trunc_error);
        cout << "LambdaB=" << lambdaB().GetDiagonal() << endl;
//        cout << "Vd=" << Vd.GetLimits() << endl;
//        cout << "U =" << U.GetLimits() << endl;
        auto [Vdr,Ur]=ReshapeVU(itsd,Vd,U); // move physical indices from inside to outside.
//        cout << "Vdr=" << Vdr.GetLimits() << endl;
//        cout << "Ur =" << Ur.GetLimits() << endl;
        DiagonalMatrixRT lB=Extend(lambdaB(),itsd);
//        cout << "lB=" << lB << endl;
        if (Min(lambdaA())<1e-13)
        {
            // Did we forget to remove smalle SVs on the previous call??
            Logger->LogWarnV(0,"iTEBDStateImp::Unpack_iMPO small lambdaA min(lambda)=%.1e", Min(lambdaA()) );
            cout << "LambdaA=" << lambdaA().GetDiagonal() << endl;
        }
        DiagonalMatrixRT lainv=Extend(1.0/lambdaA(),Dw); //extended inverse of LambdaA
        // If we just resized (increased D) then some of the lambdaA's could be zero.
        for (int i=1;i<=lainv.GetDiagonal().size();i++)
            if (isinf(lainv(i,i)))
                lainv(i)=0.0;
//        cout << "lainv=" << lainv.GetDiagonal() << endl;

        theta_AB=lB*Vdr*lainv*Ur*lB;
//        theta_AB=Vdr*Ur;
//        cout << "theta_AB=" << theta_AB.GetLimits() << endl;
    }
    auto [U,s,Vd]=svd_solver->SolveAll(theta_AB,1e-13);
    double svdError=Max(fabs(U*s*Vd-theta_AB));
    if (svdError>theta_AB.GetNumRows()*theta_AB.GetNumCols()*1e-13)
    {
        Logger->LogWarnV(0,"iTEBDStateImp::Unpack_iMPO large #2 SVD error=%.1e",svdError  );
    }
//    cout << s.GetDiagonal() << endl;
    double trunc_error=D_compressor->Compress(U,s,Vd);
//    cout << s.GetDiagonal() << endl;
    s1.bondA->SetSingularValues(s,trunc_error);
    cout << "LambdaA=" << lambdaA().GetDiagonal() << endl;
    delete svd_solver;
//    cout << "U,Vd=" << U.GetLimits() << " " << Vd.GetLimits() << endl;

    int D1=U.GetNumRows()/itsd;
    int D2=U.GetNumCols();
//    cout << "New dimensions = " << D1 << " " << D2 << endl;
//    NewBondDimensions(D1,D2);
    if (Min(lambdaB())<1e-13)
    {
        // Did we forget to remove smalle SVs on the previous call??
        Logger->LogWarnV(0,"iTEBDStateImp::Unpack_iMPO small lambdaB min(lambda)=%.1e", Min(lambdaB()) );
        cout << "LambdaB=" << lambdaB().GetDiagonal() << endl;
    }
    DiagonalMatrixRT lbinv=1.0/lambdaB(); //inverse of LambdaB
    for (int n=0; n<itsd; n++)
        for (int i=1; i<=D1; i++)
        for (int j=1; j<=D2; j++)
        {
            GammaA()[n](i,j)=lbinv(i)*U (n*D1+i,     j);
            GammaB()[n](j,i)=         Vd(     j,n*D1+i)*lbinv(i);
//            GammaA()[n](i,j)=U (n*D1+i,     j);
//            GammaB()[n](j,i)=Vd(     j,n*D1+i);
        }

}


iTEBDStateImp::MdType iTEBDStateImp::GetEigenMatrix(TensorNetworks::Direction lr, int D, const Matrix4CT& theta)
{
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
                Logger->LogWarnV(2,"iTEBDStateImp::GetEigenMatrix Dominant left eigenvalue has large imaginary component e=(%.6f,%.1e)",real(e(1)),imag(e(1)));
            eigenValue=real(e(1));
            eigenVector=U.GetColumn(1);
            break;
        }
       case DRight:
        {
            auto [U,e]=solver->SolveRightNonSym(theta.Flatten(),1e-13,1);
            if (fabs(imag(e(1)))>1e-13)
                Logger->LogWarnV(2,"iTEBDStateImp::GetEigenMatrix Dominant right eigenvalue has large imaginary component e=(%.6f,%.1e)",real(e(1)),imag(e(1)));
            eigenValue=real(e(1));
            eigenVector=U.GetColumn(1);
            break;
        }
    }
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
    if (err>1e-13)
        Logger->LogWarnV(2,"iTEBDStateImp::GetEigenMatrix large eigen error=%.1e",err);

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
        Logger->LogWarnV(1,"iTEBDStateImp::Factor small eigenvalue min(e)==%.1e", Min(e));
        cout << "e=" << e << endl;
    }

    Xinv=DiagonalMatrix<double>(1.0/sqrt(e))*Transpose(conj(U));
    //assert(IsUnit(X*Xinv,1e-13));
    return std::make_tuple(X,Xinv);
}

//
// Iterative version Ho N. Phien, Ian P. McCulloch, and Guifré Vidal, "Fast convergence of imaginary
// time evolution tensor network algorithms by recycling the environment", Physical Review B 91, 11 (2015).
//
void iTEBDStateImp::OrthogonalizeI(dVectorT& gamma, DiagonalMatrixRT& lambda,double eps,int maxIter)
{
    int d=gamma.size();
    assert(d>0);
    int D=gamma[0].GetNumRows();
    assert(D==gamma[0].GetNumCols());
    MatrixCT Vr,Vl,I(D,D); //Right ei
    Unit(I);
    SVDSolver<dcmplx>* svd_solver=new LapackSVDSolver<dcmplx>();
    dcmplx er;
    dcmplx el;
    double deltal=0;
    int niter=0;
    Logger->LogInfoV(3,"iTEBDStateImp::OrthogonalizeI: D=%4d,eps=%.1e, maxIter=%4d",D,eps,maxIter);
//    cout << "lambda=" << lambda.GetDiagonal() << endl;
    Logger->LogInfoV(3," niter  deltal        er                  el           er-el     er-1     el-1");
    do
    {
        Vr=GetNormMatrix(DRight,gamma*lambda); //=Er*I
        Vl=GetNormMatrix(DLeft ,lambda*gamma); //=I*El
        double minVr=Min(fabs(Vr.GetDiagonal()));
        double minVl=Min(fabs(Vl.GetDiagonal()));
        double epsV=1e-10;
        if (minVr<epsV)
            Logger->LogWarnV(2,"iTEBDStateImp::OrthogonalizeI Singular Vr=%.1e < %.1e, Niter=%4d, bailing out",minVr,epsV,niter);
        if (minVl<epsV)
            Logger->LogWarnV(2,"iTEBDStateImp::OrthogonalizeI Singular Vl=%.1e < %.1e, Niter=%4d, bailing out",minVl,epsV,niter);
        if (minVr<epsV|| minVl<epsV)
            return; //Bail if V is singular. THis can happen when increasing D.
//
//  Try to normalize
//
        er=Vr(1,1);
        el=Vl(1,1);
        Vr/=er;
        Vl/=el;
        for (int n=0;n<d;n++)
            gamma[n]/=sqrt(el);
////
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
        if (niter>0)
        {
            double errx=Max(fabs(X*Xinv-I));
            double erry=Max(fabs(YT*YTinv-I));
            if (errx>D*1e-13)
                Logger->LogWarnV(1,"iTEBDStateImp::OrthogonalizeI large error in X*Xinv-I=%.1e, niter=%4d",errx,niter);
            if (erry>D*1e-13)
                Logger->LogWarnV(1,"iTEBDStateImp::OrthogonalizeI large error in YT*YTinv-I=%.1e, niter=%4d",erry,niter);
            assert(IsUnit(X *Xinv ,D*1e-10));
            assert(IsUnit(YT*YTinv,D*1e-10));
        }
        if (isnan(X))
        {
            Logger->LogWarnV(1,"iTEBDStateImp::OrthogonalizeI Singular Vr Niter=%4d, bailing out",niter);
            cout << "lambda=" << lambda.GetDiagonal() << endl;
            return; //Bail out
        }
        if (isnan(Yd))
        {
            Logger->LogWarnV(1,"iTEBDStateImp::OrthogonalizeI Singular Vl Niter=%4d, bailing out",niter);
            cout << "lambda=" << lambda.GetDiagonal() << endl;
            return; //Bail out
        }
        //
        //  Transform lambda and SVD
        //
        assert(!isnan(YT));
        assert(!isnan(X));
        assert(!isnan(lambda));
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
        MatrixCT VX=VT*Xinv;
        MatrixCT YU=YTinv*U;
        for (int n=0;n<d;n++)
        {
            MatrixCT gamma_prime=VX*gamma[n];
            gamma[n]=gamma_prime*YU;
        }
        deltal=Max(fabs(lambda-lambda_prime));
        lambda=lambda_prime;
        Logger->LogInfoV(3," %4d  %.1e  (%.5f,%.1e) (%.5f,%.1e)   %.1e  %.1e   %.1e"
           ,niter,deltal,real(er),imag(er),real(el),imag(el),fabs(er-el),fabs(er-1.0),fabs(el-1.0));
        if (deltal<eps) break;
        niter++;
    } while (niter<maxIter);
    if (niter==maxIter)
        Logger->LogWarnV(1,"iTEBDStateImp::OrthogonalizeI not converged after=%4d iterations, eps=%.1e, deltal=%.1e",niter,eps,deltal);

    delete svd_solver;
//
//  Normalize
//
        for (int n=0;n<d;n++)
            gamma[n]/=sqrt(er);

    //
    //  Verify orthogonaly  .. we haven't assigned anything yet
    //
    double Oerror=GetOrthonormalityErrors(gamma,lambda);
    double epsO=D*D*eps;
    if (Oerror>epsO)
        Logger->LogWarnV(3,"iTEBDStateImp::OrthogonalizeI Large orthonormaility error=%.1e > %.1e",Oerror,epsO);

}

//
// Eigen vector version as per PHYSICAL REVIEW B 78, 155117 2008
//
void iTEBDStateImp::Orthogonalize(dVectorT& gamma, DiagonalMatrixRT& lambda)
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

    auto [Vr,er]=GetEigenMatrix(DRight,D,Er);
    auto [Vl,el]=GetEigenMatrix(DLeft ,D,El);
    double minVr=Min(fabs(Vr.GetDiagonal()));
    double minVl=Min(fabs(Vl.GetDiagonal()));
    double epsV=1e-10;
    if (minVr<epsV)
        Logger->LogWarnV(2,"iTEBDStateImp::Orthogonalize Singular Vr=%.1e > %.1e, bailing out",minVr,epsV);
    if (minVl<epsV)
        Logger->LogWarnV(2,"iTEBDStateImp::Orthogonalize Singular Vl=%.1e > %.1e, bailing out",minVl,epsV);
    if (minVr<epsV || minVl<epsV)
        return; //Bail if V is singular. THis can happen when increasing D.
//
//  Normalize
//
    double epsA=2e-13;
    if (fabs(er-el)>=epsA)
        Logger->LogWarnV(2,"iTEBDStateImp::Orthogonalize large eigenvalue asymmetry er-el=%.1e > %.1e, bailing out",fabs(er-el),epsA);
    Er.Flatten()/=er;
    El.Flatten()/=el;
    for (int n=0;n<d;n++)
        gamma[n]/=sqrt(er);
//
// Check eigen matrix accuracy.
//
    double rerr=Max(fabs(Er*Vr-Vr));
    if (rerr>1e-13)
        Logger->LogWarnV(2,"iTEBDStateImp::Orthogonalize Large right eigen error=%.1e",rerr);
    assert(rerr<1e-10);

    double lerr=Max(fabs(Vl*El-Vl));
    if (lerr>1e-13)
        Logger->LogWarnV(2,"iTEBDStateImp::Orthogonalize Large left  eigen error=%.1e",lerr);
    assert(lerr<1e-10);
//
//  Make sure Vr and Vl are roughly Hermitian, Factor will try and remove assym noise.
//
    assert(IsHermitian(Vr,1e-10));
    assert(IsHermitian(Vl,1e-10));

//
//  Decompose eigen matrices
//
    auto [X,Xinv]=Factor(Vr);
    auto [Y,Yinv]=Factor(Vl);
    MatrixCT YT   =Transpose(Y);
    MatrixCT YTinv=Transpose(Yinv);

    double epsInv=1e-13;
    if(!IsUnit(X*Xinv,epsInv))
        Logger->LogWarnV(2,"iTEBDStateImp::Orthogonalize Large inversions X error=%.1e > %.1e",Max(fabs(X*Xinv-I)),epsInv);
    if(!IsUnit(YT*YTinv,epsInv))
        Logger->LogWarnV(2,"iTEBDStateImp::Orthogonalize Large inversions X error=%.1e > %.1e",Max(fabs(YT*YTinv-I)),epsInv);
    //
    //  Transform lambda and SVD
    //
    MatrixCT YlX=YT*lambda*X; //THis will scale lambda by 1/D since YT and X are both O(1/sqrt(D))
    SVDSolver<dcmplx>* svd_solver=new LapackSVDSolver<dcmplx>();
    auto [U,lambda_prime,VT]=svd_solver->SolveAll(YlX,1e-13);
    delete svd_solver;
    lambda=lambda_prime;
    //
    //  Transform Gamma
    //
    MatrixCT VX=VT*Xinv;
    MatrixCT YU=YTinv*U;
    for (int n=0;n<d;n++)
    {
        MatrixCT VXG=VX*gamma[n];
        gamma[n]=VXG*YU;
    }

    double Oerror=GetOrthonormalityErrors(gamma,lambda);
    double eps=D*1e-12;
    if (Oerror>eps)
        Logger->LogWarnV(2,"iTEBDStateImp::Orthogonalize Large orthonormaility error=%.1e > %.1e",Oerror,eps);

    Logger->LogInfoV(2,"iTEBDStateImp::Orthogonalize complete R/L orthonormaility error=%.1e",Oerror);
}


}
