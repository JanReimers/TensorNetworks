#include "TensorNetworksImp/iMPS/iMPSSite.H"
#include "TensorNetworksImp/MPS/Bond.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/CheckSpin.H"
#include "Operators/OperatorValuedMatrix.H"
#include "NumericalMethods/LapackEigenSolver.H"
#include "NumericalMethods/ArpackEigenSolver.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include "NumericalMethods/LapackLinearSolver.H"
#include "TensorNetworks/TNSLogger.H"
#include "Containers/Matrix6.H"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

namespace TensorNetworks
{

iMPSSite::iMPSSite(Bond* leftBond, Bond* rightBond,int d, int D)
    : itsLeft_Bond(leftBond)
    , itsRightBond(rightBond)
    , itsd(d)
    , itsD1(D)
    , itsD2(D)
    , itsM(d,D,D)
    , itsA(d,D,D)
    , itsB(d,D,D)
    , itsEigenSolver(new LapackEigenSolver<dcmplx>())
    , itsNumUpdates(0)
    , itsEmin(0.0)
    , itsGapE(0.0)
    , itsIterDE(1.0)
{
    assert(itsRightBond);
    assert(itsLeft_Bond);
}

iMPSSite::~iMPSSite()
{
    delete itsEigenSolver;
}

void iMPSSite::InitializeWith(State state,int sgn)
{
    itsM.InitializeWith(state,sgn);
    if (state==Random)
        itsM.QLRR(DLeft,1e-14); //Make it left normalized.  The main goal here is to just get it normalized.
}

void iMPSSite::NewBondDimensions(int D1, int D2, bool saveData)
{
    assert(false);
    assert(D1>0);
    assert(D2>0);
    if (itsD1==D1 && itsD2==D2) return;

    itsD1=D1;
    itsD2=D2;
}

bool iMPSSite::IsNormalized(Direction lr,double eps) const
{
    return IsUnit(GetNorm(lr),eps);
}

bool iMPSSite::IsCanonical(Direction lr,double eps) const
{
    return IsUnit(GetCanonicalNorm(lr),eps);
}

char iMPSSite::GetNormStatus(double eps) const
{
    char ret;

    if (IsNormalized(DLeft,eps))
    {
        if (IsNormalized(DRight,eps))
            ret='I'; //This should be rare
        else
            ret='A';
    }
    else if (IsNormalized(DRight,eps))
        ret='B';
    else
    {
        bool cl=IsCanonical(DLeft ,eps);
        bool cr=IsCanonical(DRight,eps);
        if (cl && cr)
            ret='G';
        else if (cl && !cr)
            ret='l';
        else if (cr && !cl)
            ret='r';
        else
            ret='M';
    }

    return ret;
}

void iMPSSite::Report(std::ostream& os) const
{
    os << std::setw(4)          << itsD1
       << std::setw(4)          << itsD2
//       << std::setw(5)          << GetNormStatus(1e-12)
       << std::setw(8)          << itsNumUpdates
       << std::fixed      << std::setprecision(4) << std::setw(8) << GetFrobeniusNorm()
       << std::fixed      << std::setprecision(8) << std::setw(13) << itsEmin
       << std::fixed      << std::setprecision(5) << std::setw(10) << itsGapE
       << std::scientific << std::setprecision(1) << std::setw(10) << itsIterDE
       ;
}

MatrixCT iMPSSite::GetNorm(Direction lr) const
{
    return itsM.GetNorm(lr);
}
MatrixCT iMPSSite::GetCanonicalNorm(Direction lr) const
{
    return itsM.GetNorm(lr,GetBond(lr)->GetSVs());
}

double   iMPSSite::GetFrobeniusNorm() const
{
    return 0.0;
}

void   iMPSSite::InitQRIter()
{
    assert(itsM.IsSquare()); //Make sure we are square
    itsG.SetLimits(itsM.GetLimits());
    Unit(itsG);
}
double iMPSSite::QRStep(Direction lr,double eps)
{
    double eta=99.0;
    MatrixCT L=itsM.QLRR(lr,eps); //Solves M=Q*L, Q is stored in M
    L*=1.0/L(1,1); //Force normalization as we go.
    if (L.IsSquare())
    {
        MatrixRT Id(L.GetLimits());
        Unit(Id);
        eta=Max(fabs(L-Id));
//        cout << " L=" << L << "eta=" << eta << endl;
    }
    else
    {
            cout << "Changed L=" << L.GetLimits() << endl;
    }

    GetBond(lr)->TransferQR(lr,L); //  Do M->L*M
    switch(lr)
    {
    case DLeft:
        itsG=L*itsG; //Update gauge transform
        break;
    case DRight:
        itsG=itsG*L; //Update gauge transform
        break;
    }
 //   cout << "G=" << itsG << endl;
    return eta;
}

void iMPSSite::SaveAB_CalcLR(Direction lr)
{
    dcmplx norm=Trace(itsG*~itsG);
    assert(fabs(std::imag(norm))<1e-14);
    itsG/=sqrt(std::real(norm));
    switch(lr)
    {
    case DLeft:
        itsA=itsM;
        break;
    case DRight:
        itsB=itsM;
        break;
    }
}

Matrix6CT ContractHAC(const MatrixOR& W, const Tensor3& LW, const Tensor3& RW)
{
    auto [Dw,D1,D2]=LW.GetDimensions();
    auto [X1,X2]=W.GetChi12();
    assert(X1==X2);
    assert(Dw==X1+2);
    int d=W.Getd();
    Matrix6CT Hac(d,D1,D1,d,D2,D2,0); //zero index for m,n
//    OperatorZ Z(d);

    for (int m=0;m<d;m++)
    for (int n=0;n<d;n++)
        for (index_t i1=1;i1<=D1;i1++)
        for (index_t i2=1;i2<=D2;i2++)
            for (index_t j1=1;j1<=D1;j1++)
            for (index_t j2=1;j2<=D2;j2++)
            {
                dcmplx t(0.0);
                for (index_t w1=1;w1<=Dw;w1++)
                for (index_t w2=1;w2<=Dw;w2++)
                    t+=W(w1-1,w2-1)(m,n)*LW(w1)(i1,i2)*RW(w2)(j1,j2);
                Hac(m,i1,j1,n,i2,j2)=t;
            }
    return Hac;
}

Matrix4CT ContractHC(const Tensor3& LW, const Tensor3& RW)
{
    auto [Dw,D1,D2]=LW.GetDimensions();
    Matrix4CT Hc(D1,D1,D2,D2); //zero index for m,n

    for (index_t i1=1;i1<=D1;i1++)
    for (index_t i2=1;i2<=D2;i2++)
        for (index_t j1=1;j1<=D1;j1++)
        for (index_t j2=1;j2<=D2;j2++)
        {
            dcmplx t(0.0);
            for (index_t w=1;w<=Dw;w++)
                t+=LW(w)(i1,i2)*RW(w)(j1,j2);
            Hc(i1,j1,i2,j2)=t;
        }
    return Hc;
}

double iMPSSite::Refine (const SiteOperator* h,const Epsilons& eps)
{
    double epsHerm=5e-8;
    const MatrixOR& W=h->GetW();
    auto [d,D1,D2]=itsA.GetDimensions();
    auto [el,LW]=itsA.GetLW(W);
    auto [er,RW]=itsB.GetRW(W);
    double e=0.5*std::real(el+er);
    itsIterDE=e-itsEmin;
    itsEmin=e;

    Matrix6CT Hac=ContractHAC(W,LW,RW);
    MatrixCT  Hacf=Hac.Flatten();
    //cout << "Hac=" << Hac << endl;
    assert(IsHermitian(Hacf,epsHerm));
    Matrix4CT Hc=ContractHC(LW,RW);
    MatrixCT  Hcf=Hc.Flatten();
    //cout << "Hc=" << Hc << endl;
    assert(IsHermitian(Hcf,epsHerm));

    Hacf=0.5*(Hacf+~Hacf);
    Hcf =0.5*(Hcf +~Hcf );
    LapackEigenSolver<dcmplx> asolver;
    auto [Acf,eAc]=asolver.Solve(Hacf,1e-12,1);
    auto [Cf ,eC ]=asolver.Solve(Hcf ,1e-12,1);

    Tensor3 Ac(d,D1,D2);
    Ac.UnFlatten(VectorCT(Acf.GetColumn(1)));
    MatrixCT C=UnFlatten(VectorCT(Cf.GetColumn(1)));
    MatrixCT Cdagger=~C;
    cout << std::fixed;
    //cout << "Ac=" << Ac << endl;
    //cout << "eAc=" << eAc << endl;
    //cout << "C=" << C << endl;
   // cout << "eC=" << eC << endl;

    LapackSVDSolver<dcmplx> SVDsolver;
    {
        auto [U,s,VT]=SVDsolver.SolveAll(C,1e-14);
        //cout << std::scientific << "C svs=" << s.GetDiagonal() << endl;
        //cout << std::scientific << "C min s=" << Min(s.GetDiagonal()) << endl;
        assert(Min(s.GetDiagonal())>1e-14);
        double s2=s.GetDiagonal()*s.GetDiagonal();
        //cout << "s*s-1=" << s2-1.0 << endl;
        assert(fabs(s2-1.0)<1e-13);
        itsRightBond->SetSingularValues(s,0.0);
    }

    MatrixCT AcL=Ac.Flatten(DLeft);
    MatrixCT AcR=Ac.Flatten(DRight);
    MatrixCT AcC=AcL*Cdagger;
    MatrixCT CAc=Cdagger*AcR;
    double etaL,etaR;
    Tensor3 Anew(d,D1,D2),Bnew(d,D1,D2);
    {
        auto [Ul,sl,VTl]=SVDsolver.SolveAll(AcC,1e-14);
        auto [Ur,sr,VTr]=SVDsolver.SolveAll(CAc,1e-14);
        //cout << "Ul*Vl=" << Ul*VTl << endl;
        //cout << "Ur*Vr=" << Ur*VTr << endl;
        Anew.UnFlatten(DLeft ,Ul*VTl);
        Bnew.UnFlatten(DRight,Ur*VTr);
        etaL=FrobeniusNorm(AcL-Ul*VTl*C);
        etaR=FrobeniusNorm(AcR-C*Ur*VTr);
    }
    //cout << "Anew=" << Anew << endl;
    //cout << "Bnew=" << Bnew << endl;

    itsM=Ac;
    itsA=Anew;
    itsB=Bnew;
    itsG=C;

    double etaG=0.0;
    for (int n=0;n<d;n++)
        etaG+=FrobeniusNorm(itsA(n)*C-C*itsB(n));
    etaG/=d;

//    cout << std::fixed << std::setprecision(12) << "E=" << itsEmin;
//    cout << std::scientific << std::setprecision(1) << " DE=" << itsIterDE << "  etaL,etaR,etaG=" << etaL << " " << etaR << " " << etaG << endl;

    //cout << "A.Norm=" << itsA.GetNorm(DLeft) << endl;
    //cout << "B.Norm=" << itsB.GetNorm(DRight) << endl;
    return std::max(etaL,etaR);
}

double iMPSSite::GetExpectation(const SiteOperator* so) const
{

    assert(so);
    MatrixOR W=so->GetW();
    assert(W.GetForm()==RegularLower);
    assert(IsLowerTriangular(W));
    return itsA.GetExpectation(W);

    /*
//    auto [d,D1,D2]=itsA.GetDimensions();
    const OpRange& wr=so->GetRanges();
    int Dw=wr.Dw1;
    assert(Dw==wr.Dw2);
//
//  Fill out E[5]
//
    dVectorT E(Dw+1);
    E[Dw]=MatrixCT(D1,D2);
    Unit(E[Dw]);
//
//  Check E[Dw] is self consistent
//
    MatrixCT EDw(D1,D2);
    Fill(EDw,dcmplx(0));
    for (int m=0; m<d; m++)
        for (int n=0; n<d; n++)
        {
            for (int i2=1;i2<=D2;i2++)
            for (int j2=1;j2<=D2;j2++)
                for (int i1=1;i1<=D1;i1++)
                for (int j1=1;j1<=D1;j1++)
                    EDw(i2,j2)+=conj(itsA(m)(i1,i2))*W(Dw-1,Dw-1)(m,n)*E[Dw](i1,j1)*itsA(n)(j1,j2);
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
        MatrixCT C(D2,D2);
        Fill(C,dcmplx(0));
        for (int m=0; m<d; m++)
            for (int n=0; n<d; n++)
            {
                if (W(w1-1,w1-1)(m,n)!=0.0) //Make sure there is nothing on the diagonal.
                {
//                    std::cerr << "diagonal W["<< m << "," << n << "](" << w1 << "," << w1 << ")=" << W(w1-1,w1-1)(m,n)<< std::endl;
                    assert(m==n); //should be only unit ops on the diagonal
//                    assert(Wmn(w1,w1)==1.0);
                    diagonals.push_back(w1);
                }
                for (int w2=w1+1;w2<=Dw;w2++)
                if (W(w2-1,w1-1)(m,n)!=0.0)
                {
                    for (int i2=1;i2<=D2;i2++)
                    for (int j2=1;j2<=D2;j2++)
                        for (int i1=1;i1<=D1;i1++)
                        for (int j1=1;j1<=D1;j1++)
                            C(i2,j2)+=conj(itsA(m)(i1,i2))*W(w2-1,w1-1)(m,n)*E[w2](i1,j1)*itsA(n)(j1,j2);
                }
            }

        if (diagonals.size()==0)
        {
            E[w1]=C;
//            cout << std::fixed << "E[" << w1 << "]=" << E[w1] << endl;
        }
        else
        {

        assert(false);

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
        E[1]=MatrixCT(D2,D2);
        Fill(E[1],dcmplx(0));
        for (int w=2;w<=Dw;w++)
        {
            for (int m=0; m<d; m++)
                for (int n=0; n<d; n++)
                {
                    //cout << "W" << m << n << "=" << Wmn << endl;
                    for (int i2=1;i2<=D2;i2++)
                    for (int j2=1;j2<=D2;j2++)
                        for (int i1=1;i1<=D1;i1++)
                        for (int j1=1;j1<=D1;j1++)
                            E[1](i2,j2)+=W(w-1,0)(m,n)*conj(itsA(m)(i1,i2))*E[w](i1,j1)*itsA(n)(j1,j2);
                }
        }

    }

//  E[1] should now be the same as C in the paper.
//    cout << "E[" << 1 << "]=" << E[1] << endl;

    const DiagonalMatrixRT& s=itsRightBond->GetSVs();
//
//  Take the trace of E1
//
    dcmplx E0=0.0;
    DiagonalMatrixRT ro=s*s;
    assert(fabs(Sum(ro)-1.0)<1e-13); //Make sure we are normalized
//    MatrixCT CC=itsG*~itsG;
//    MatrixCT CC=~itsG*itsG;
//    for (int i1=1;i1<=D1;i1++)
    for (int i2=1;i2<=D2;i2++)
        E0+=E[1](i2,i2)*ro(i2,i2); //We only use the diagonal elements of E[1]
    //E0/=2.0; //Convert from energy per unit cell to energy per site.

    if (fabs(imag(E0))>=1e-10)
        Logger->LogWarnV(0,"iTEBDStateImp::GetExpectation(iMPO) E0=(%.5f,%.1e) has large imaginary component",real(E0),imag(E0));
    return real(E0);
*/
}


void iMPSSite::SVDTransfer(Direction lr,const DiagonalMatrixRT& s, const MatrixCT& UV)
{
    assert(false);
}
void iMPSSite::SVDTransfer(Direction lr,const MatrixCT& UV)
{
    assert(false);

}
void iMPSSite::TransferQR (Direction lr,const MatrixCT& G)
{
    itsM.Multiply(lr,G);
}

void iMPSSite::NormalizeQR  (Direction lr)
{
    assert(false);

}

void iMPSSite::NormalizeSVD (Direction lr,SVCompressorC*)
{
    assert(false);

}


} //namespace
