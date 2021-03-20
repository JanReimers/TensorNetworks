#include "TensorNetworksImp/iMPS/iMPSSite.H"
#include "TensorNetworksImp/MPS/Bond.H"
#include "TensorNetworks/iHamiltonian.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/CheckSpin.H"
#include "Operators/OperatorValuedMatrix.H"
#include "NumericalMethods/LapackEigenSolver.H"
#include "NumericalMethods/ArpackEigenSolver.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include "NumericalMethods/LapackLinearSolver.H"
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

    for (int n=0; n<itsd; n++)
    {
        itsMs.push_back(MatrixCT(D,D));
        Fill(itsMs.back(),std::complex<double>(0.0));
    }
}

iMPSSite::~iMPSSite()
{
    delete itsEigenSolver;
}

void iMPSSite::InitializeWith(State state,int sgn)
{
    itsM.InitializeWith(state,sgn);
    switch (state)
    {
    case Product :
    {
        int i=1;
        for (dIterT id=itsMs.begin(); id!=itsMs.end(); id++,i++)
            if (i<=itsD2)
                (*id)(1,i)=std::complex<double>(sgn); //Left normalized
        break;
    }
    case Random :
    {
        for (dIterT id=itsMs.begin(); id!=itsMs.end(); id++)
        {
            FillRandom(*id);
            (*id)*=1.0/sqrt(itsd*itsD1*itsD2); //Try and keep <psi|psi>~O(1)
        }
        break;
    }
    case Neel :
    {
        for (dIterT id=itsMs.begin(); id!=itsMs.end(); id++)
            Fill(*id,dcmplx(0.0));
        if (sgn== 1)
            itsMs[0     ](1,1)=1.0;
        if (sgn==-1)
            itsMs[itsd-1](1,1)=1.0;

        break;
    }
    }
}

void iMPSSite::NewBondDimensions(int D1, int D2, bool saveData)
{
    assert(D1>0);
    assert(D2>0);
    if (itsD1==D1 && itsD2==D2) return;
    for (int in=0; in<itsd; in++)
    {
        itsMs[in].SetLimits(D1,D2,saveData);
        for (int i1=itsD1;i1<=D1;i1++)
            for (int i2=itsD2;i2<=D2;i2++)
                itsMs[in](i1,i2)=dcmplx(0.0);
//                itsMs[in](i1,i2)=OMLRand<dcmplx>()*0.001; //If you compress to remove zero SVs in the right placese you should not need this trick
    }
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
    if (L.IsSquare())
    {
        MatrixRT Id(L.GetLimits());
        Unit(Id);
        eta=Max(fabs(L-Id));
 //           cout << " L=" << L.GetLimits() << "eta=" << eta << endl;
    }
    else
    {
//            cout << " L=" << L.GetLimits() << endl;
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
    return eta;
}

void iMPSSite::SaveAB_CalcLR(Direction lr)
{
    switch(lr)
    {
    case DLeft:
        itsA=itsM;
        itsR=itsA.GetTMEigenVector(lr);
        break;
    case DRight:
        itsB=itsM;
        itsL=itsB.GetTMEigenVector(lr);
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

void iMPSSite::Refine (const iHamiltonian* H,const Epsilons& eps)
{
    const SiteOperator* so=H->GetSiteOperator(1);
    const MatrixOR& W=so->GetW();
    cout << "W=" << W << endl;
    auto [d,D1,D2]=itsA.GetDimensions();
    auto [X1,X2]=W.GetChi12();
    assert(X1==X2);
    int Dw=X1+2;
    Matrix6CT TWL=itsA.GetTransferMatrix(DLeft ,W);
    Matrix6CT TWR=itsB.GetTransferMatrix(DRight,W);
    MatrixCT  TL=itsA.GetTransferMatrix(DLeft );
    MatrixCT  TR=itsB.GetTransferMatrix(DRight);
    assert(Max(fabs(TL-TWL.SubMatrix(1 ,1 ).Flatten()))<1e-15);
    assert(Max(fabs(TL-TWL.SubMatrix(Dw,Dw).Flatten()))<1e-15);
    assert(Max(fabs(TR-TWR.SubMatrix(1 ,1 ).Flatten()))<1e-15);
    assert(Max(fabs(TR-TWR.SubMatrix(Dw,Dw).Flatten()))<1e-15);
    Tensor3 LW(Dw,D1,D2,1);
    Tensor3 RW(Dw,D1,D2,1);
    LW.Unit(Dw);
    RW.Unit(1);
    cout << std::setprecision(3);
    for (int w=Dw-1;w>1;w--)
    {
        assert(Max(fabs(TWL.SubMatrix(w,w).Flatten()))==0.0);
        assert(Max(fabs(TWR.SubMatrix(w,w).Flatten()))==0.0);
        for (int w1=w+1;w1<=Dw;w1++)
            LW(w)+=LW(w1)*TWL.SubMatrix(w1,w);
        for (int w2=1;w2<w;w2++)
            RW(w)+=TWR.SubMatrix(w,w2)*RW(w2);
        assert(IsHermitian(LW(w),1e-15));
        assert(IsHermitian(RW(w),1e-15));
//        cout << "LW(" << w << ")=" << LW(w) << endl;
//        cout << "RW(" << w << ")=" << RW(w) << endl;
    }
    //
    //  Now we need YL_1 and YR_Dw
    //
    MatrixCT YL(D1,D2),YR(D1,D2);
    Fill(YL,dcmplx(0.0));
    Fill(YR,dcmplx(0.0));
    for (int w1=2;w1<=Dw;w1++)
        YL+=LW(w1)*TWL.SubMatrix(w1,1);
    for (int w2=1;w2< Dw;w2++)
        YR+=TWR.SubMatrix(Dw,w2)*RW(w2);

    assert(IsHermitian(YL,1e-15));
    assert(IsHermitian(YR,1e-15));
//    cout << "YL=" << YL << endl;
//    cout << "YR=" << YR << endl;
    VectorCT YLf=Flatten(YL);
    VectorCT YRf=Flatten(YR);

    assert(TL.GetLimits()==TR.GetLimits()); //should both be D1^2 x D2^2
    MatrixCT I(TL.GetLimits()),IL(D1,D1),IR(D2,D2);
    Unit(I),Unit(IL);Unit(IR);
    VectorCT ILf=Flatten(IL);
    VectorCT IRf=Flatten(IR);
//    cout << "ILf=" << ILf << endl;
//    cout << "IRf=" << IRf << endl;


    MatrixCT PL=OuterProduct(itsR,ILf);
    MatrixCT PR=OuterProduct(IRf,itsL);
//    cout << "PL=" << PL << endl;
//    cout << "PR=" << PL << endl;

    MatrixCT XL=I-TL+PL;
    MatrixCT XR=I-TR+PR;
//    cout << "XL=" << XL << endl;
//    cout << "XR=" << XR << endl;

    VectorCT vL=YLf-YLf*PL;
    VectorCT vR=YRf-PR*YRf;
//    cout << "vL=" << vL << endl;
//    cout << "vR=" << vR << endl;

    LapackSVDSolver<dcmplx> SVDsolver;
    {
        auto [U,s,VT]=SVDsolver.SolveAll(XL,1e-14);
        assert(Min(s.GetDiagonal())>1e-14);
    }
    {
        auto [U,s,VT]=SVDsolver.SolveAll(XR,1e-14);
        assert(Min(s.GetDiagonal())>1e-14);
    }

    LapackLinearSolver<dcmplx> solver;
    VectorCT LW1f=solver.Solve(vL,XL);
    VectorCT RWDf=solver.Solve(XR,vR);
    //cout << "LW1f=" << LW1f << endl;
    //cout << "RWDf=" << RWDf << endl;
    LW(1 )=UnFlatten(LW1f);
    RW(Dw)=UnFlatten(RWDf);
    for (int w=1;w<=Dw;w++)
    {
        //cout << "LW(" << w << ")=" << LW(w) << endl;
        cout << std::scientific << Max(fabs(LW(w)-~LW(w))) << endl;
        assert(IsHermitian(LW(w),1e-13));
    }
    for (int w=1;w<=Dw;w++)
    {
        //cout << "RW(" << w << ")=" << RW(w) << endl;
        cout << std::scientific << Max(fabs(RW(w)-~RW(w))) << endl;
        assert(IsHermitian(RW(w),1e-13));
    }

    Matrix6CT Hac=ContractHAC(W,LW,RW);
    //cout << "Hac=" << Hac << endl;
    assert(IsHermitian(Hac.Flatten(),1e-13));
    Matrix4CT Hc=ContractHC(LW,RW);
    //cout << "Hc=" << Hc << endl;
    assert(IsHermitian(Hc.Flatten(),1e-13));

    LapackEigenSolver<dcmplx> asolver;
    auto [Ac,eAc]=asolver.Solve(Hac.Flatten(),1e-12,1);
    auto [Cf ,eC ]=asolver.Solve(Hc .Flatten(),1e-12,1);

    MatrixCT C=UnFlatten(VectorCT(Cf.GetColumn(1)));
    cout << std::fixed;
    //cout << "Ac=" << Ac << endl;
    cout << "eAc=" << eAc << endl;
    cout << "C=" << C << endl;
    cout << "eC=" << eC << endl;

    {
        auto [U,s,VT]=SVDsolver.SolveAll(C,1e-14);
        cout << std::scientific << "C svs=" << s.GetDiagonal() << endl;
        assert(Min(s.GetDiagonal())>1e-14);
        double s2=s.GetDiagonal()*s.GetDiagonal();
        cout << "s*s-1=" << s2-1.0 << endl;
        assert(fabs(s2-1.0)<1e-13);
    }
}


void iMPSSite::SVDTransfer(Direction lr,const DiagonalMatrixRT& s, const MatrixCT& UV)
{

}
void iMPSSite::SVDTransfer(Direction lr,const MatrixCT& UV)
{

}
void iMPSSite::TransferQR (Direction lr,const MatrixCT& R)
{

}

void iMPSSite::NormalizeQR  (Direction lr)
{

}

void iMPSSite::NormalizeSVD (Direction lr,SVCompressorC*)
{

}


} //namespace
