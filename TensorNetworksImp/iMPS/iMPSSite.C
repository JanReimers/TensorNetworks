#include "TensorNetworksImp/iMPS/iMPSSite.H"
#include "TensorNetworksImp/MPS/Bond.H"
#include "TensorNetworks/iHamiltonian.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/CheckSpin.H"
#include "Operators/OperatorValuedMatrix.H"
#include "NumericalMethods/LapackEigenSolver.H"
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

void iMPSSite::Refine (const iHamiltonian* H,const Epsilons& eps)
{
    const SiteOperator* so=H->GetSiteOperator(1);
    const MatrixOR& W=so->GetW();
    cout << "W=" << W << endl;
//    itsA=itsM;
//    itsB=itsM;
    cout << std::setprecision(3);
    Matrix6CT TL=itsA.GetTransferMatrix(DLeft ,W);
    Matrix6CT TR=itsB.GetTransferMatrix(DRight,W);
    cout << "TA=" << itsA.GetTransferMatrix(DLeft);
    cout << "TL(1,1)=" << TL.SubMatrix(1,1) << endl;
    cout << "TB=" << itsB.GetTransferMatrix(DRight);
    cout << "TR(1,1)=" << TR.SubMatrix(1,1) << endl;
    auto [d,D1,D2]=itsA.GetDimensions();
    auto [X1,X2]=W.GetChi12();
    assert(X1==X2);
    int Dw=X1+2;
    Tensor3 LW(Dw,D1,D2,1);
    Tensor3 RW(Dw,D1,D2,1);
    LW.Unit(Dw);
    RW.Unit(1);
    for (int w=Dw-1;w>1;w--)
    {
        Matrix4CT TLww=TL.SubMatrix(w,w);
//        cout << "TL(" << w << "," << w << ")=" << TLww << endl;
        assert(Max(fabs(TLww.Flatten()))==0.0);
        Matrix4CT TRww=TR.SubMatrix(w,w);
//        cout << "TR(" << w << "," << w << ")=" << TRww << endl;
        assert(Max(fabs(TRww.Flatten()))==0.0);
        for (int w1=w+1;w1<=Dw;w1++)
            LW(w)+=LW(w1)*TL.SubMatrix(w1,w);
        for (int w2=1;w2<w;w2++)
            RW(w)+=TR.SubMatrix(w,w2)*RW(w2);
        cout << "LW(" << w << ")=" << LW(w) << endl;
        cout << "RW(" << w << ")=" << RW(w) << endl;
    }
    cout << "TR(" << 2 << "," << 1 << ")=" << TR.SubMatrix(2,1)<< endl;
    cout << "TL(" << 3 << "," << 2 << ")=" << TL.SubMatrix(3,2)<< endl;
    //
    //  Now we need YL_1 and YR_Dw
    //
    MatrixCT YL(D1,D2),YR(D1,D2);
    Fill(YL,dcmplx(0.0));
    Fill(YR,dcmplx(0.0));
    for (int w1=2;w1<=Dw;w1++)
        YL+=LW(w1)*TL.SubMatrix(w1,1);
    for (int w2=1;w2< Dw;w2++)
        YR+=TR.SubMatrix(Dw,w2)*RW(w2);
    cout << "YL=" << YL << endl;
    cout << "TL=" <<TL.SubMatrix(1,1) << endl;
    cout << "YR=" << YR << endl;
    cout << "TR=" <<TR.SubMatrix(Dw,Dw) << endl;

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
