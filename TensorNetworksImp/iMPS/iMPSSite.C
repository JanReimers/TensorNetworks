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

iMPSSite::iMPSSite(Bond* leftBond, Bond* rightBond,int d, int D,int siteNumber)
    : itsLeft_Bond(leftBond)
    , itsRightBond(rightBond)
    , itsd(d)
    , itsD1(D)
    , itsD2(D)
    , itsSiteNumber(siteNumber)
    , itsM(d,D,D,"M",itsSiteNumber)
    , itsA(d,D,D,"A",itsSiteNumber)
    , itsB(d,D,D,"B",itsSiteNumber)
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
    if (state==Random || state==Constant)
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

double iMPSSite::GetANormError() const
{
    MatrixCT An=itsA.GetNorm(DLeft);
    MatrixCT I(An.GetLimits());
    Unit(I);
    return ::FrobeniusNorm(An-I);
}
double iMPSSite::GetBNormError() const
{
    MatrixCT Bn=itsB.GetNorm(DRight);
    MatrixCT I(Bn.GetLimits());
    Unit(I);
    return ::FrobeniusNorm(Bn-I);
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
        //itsG=itsG*L; //Update gauge transform
        break;
    }
 //   cout << "G=" << itsG << endl;
    return eta;
}

void iMPSSite::TransferQR (Direction lr,const MatrixCT& G)
{
    itsM.Multiply(lr,G);
    if (lr==DRight) itsG=itsG*G; //Update gauge transform
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
        itsA.SetLabel("A");
        break;
    case DRight:
        itsB=itsM;
        itsB.SetLabel("B");
        break;
    }
}

//
//  ||A(k)*G(k)-G(k-1)B(k)||
//
double iMPSSite::GetGaugeError(const iMPSSite* left_neighbour) const
{
    int d=itsA.Getd();
    double etaG=0.0;
    for (int n=0;n<d;n++)
        etaG+=FrobeniusNorm(itsA(n)*itsG-left_neighbour->itsG*itsB(n));
    return etaG;
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

double iMPSSite::RefineOneSite (const MatrixOR& W,const Epsilons& eps)
{
    double epsHerm=5e-8;
    auto [d,D1,D2]=itsA.GetDimensions();
//
//  Check gauge transform
//
//    double etaG1=0.0;
//    for (int n=0;n<d;n++)
//    {
////        cout << itsA(n)*itsG-itsG*itsB(n) << endl;
//        etaG1+=FrobeniusNorm(itsA(n)*itsG-itsG*itsB(n));
//    }
//    etaG1/=d;
//    cout << std::scientific << "etaG=" << etaG1 << endl;
//    assert(etaG1<1e-14);

    VectorCT  L=TensorNetworks::Flatten(Transpose(~itsG*itsG));
    VectorCT  R=TensorNetworks::Flatten(Transpose(itsG*~itsG));
    auto [el,LW]=itsA.GetLW(W,R);
    auto [er,RW]=itsB.GetRW(W,L);
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

    Tensor3 Ac(d,D1,D2,"Ac",itsSiteNumber);
    Ac.UnFlatten(VectorCT(Acf.GetColumn(1)));
    MatrixCT C=UnFlatten(VectorCT(Cf.GetColumn(1)));
    MatrixCT Cdagger=~C;
    cout << std::fixed;
    //cout << "Ac=" << Ac << endl;
    cout << "eM=" << eAc << endl;
    //cout << "C=" << C << endl;
    cout << "eC=" << eC << endl;

    VectorRT sC;
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
        sC=s.GetDiagonal();
    }

    MatrixCT AcL=Ac.Flatten(DLeft);
    MatrixCT AcR=Ac.Flatten(DRight);
    MatrixCT AcC=AcL*Cdagger;
    MatrixCT CAc=Cdagger*AcR;
    double etaL,etaR;
    Tensor3 Anew(d,D1,D2,"A",itsSiteNumber),Bnew(d,D1,D2,"B",itsSiteNumber);
    {
        auto [Ul,sl,VTl]=SVDsolver.SolveAll(AcC,1e-14);
        auto [Ur,sr,VTr]=SVDsolver.SolveAll(CAc,1e-14);
        cout << "sl=" << sl.GetDiagonal() << endl;
        cout << "sr=" << sr.GetDiagonal() << endl;
        cout << "sl norm=" << sl.GetDiagonal()*sl.GetDiagonal();
        cout << "sr norm=" << sr.GetDiagonal()*sr.GetDiagonal() << endl;
        //cout << "Ul*Vl=" << Ul*VTl << endl;
        //cout << "Ur*Vr=" << Ur*VTr << endl;
        Anew.UnFlatten(DLeft ,Ul*VTl);
        Bnew.UnFlatten(DRight,Ur*VTr);
        etaL=FrobeniusNorm(AcL-Ul*VTl*C);
        etaR=FrobeniusNorm(AcR-C*Ur*VTr);
        VectorRT sC2=DirectMultiply(sC,sC); //sC^2
        cout << "sl-sC2=" << sl.GetDiagonal()-sC2 << endl;
        cout << "sr-sC2=" << sr.GetDiagonal()-sC2 << endl;
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

RefineData iMPSSite::Refine(const MatrixOR& W,const MatrixOR& Wcell,iMPSSite* left_neighbour,const UnitcellMPSType& AB,const Epsilons& eps)
{
//    cout << "Refine itsM,itsA,itsB=" << itsM.GetLabel() << " " <<  itsA.GetLabel() << " " <<  itsB.GetLabel() << endl;

    RefineData rd; //Struct for storing all the errors
    auto [d,D1,D2]=itsA.GetDimensions();
//
//  Check gauge transform
//
//    double etaG1=0.0,etaG2=0.0;
//    for (int n=0;n<d;n++)
//    {
////        cout << itsA(n)*itsG-left_neighbour->itsG*itsB(n) << endl;
//        etaG1+=FrobeniusNorm(itsA(n)*itsG-left_neighbour->itsG*itsB(n));
//        etaG2+=FrobeniusNorm(left_neighbour->itsA(n)*left_neighbour->itsG-itsG*left_neighbour->itsB(n));
//    }
////    etaG1/=d;
//    cout << std::setprecision(4) << std::scientific << "etaG1, etaG2==" << etaG1 << " " << etaG2 << endl;
//    assert(etaG1<1e-14);    //left_neighbour=this;
    double epsHerm=5e-8;
    VectorCT  L=TensorNetworks::Flatten(Transpose(~itsG*itsG));
    VectorCT  R=TensorNetworks::Flatten(Transpose(itsG*~itsG));
    VectorCT  Lm=TensorNetworks::Flatten(Transpose(~(left_neighbour->itsG)*left_neighbour->itsG));
    VectorCT  Rp=TensorNetworks::Flatten(Transpose(left_neighbour->itsG*~(left_neighbour->itsG)));
    auto [A,Ap,B,Bm]=AB;
//    cout << "A,Ap,B,Bm=" << A.GetLabel() << " " <<  Ap.GetLabel() << " " <<  B.GetLabel() << " " <<  Bm.GetLabel() << endl;

//    VectorCT  Rp=Ap.GetTMEigenVector(DLeft );
//    VectorCT  L =B .GetTMEigenVector(DRight);
//    VectorCT  R =A .GetTMEigenVector(DLeft );
//    VectorCT  Lm=Bm.GetTMEigenVector(DRight);

//    const MatrixOR& W=h->GetW();
    auto [el ,LW ]=A .GetLW(Wcell,R);
    auto [er ,RW ]=B .GetRW(Wcell,L);
    auto [elp,LWp]=Ap.GetLW(Wcell,Rp);
    auto [erm,RWm]=Bm.GetRW(Wcell,Lm); //Should be Lmm=L(k-2) for L>2
//    cout << "LW,RW,LWp,RWm=" << LW.GetLabel() << " " <<  RW.GetLabel() << " " <<  LWp.GetLabel() << " " <<  RWm.GetLabel() << endl;

    rd.eL=el;
    rd.eR=er;
    rd.eLp=elp;
    rd.eRm=erm;
//    double e=0.5*std::real(el+er); //one always seems to be way off
    double e=std::min(std::real(el),std::real(er));
    itsIterDE=e-itsEmin;
    itsEmin=e;

    Matrix6CT Hac=ContractHAC(W,LW,RW);
    MatrixCT  Hacf=Hac.Flatten();
    assert(IsHermitian(Hacf,epsHerm));

    Matrix4CT HcR=ContractHC(LWp,RW);
    MatrixCT  HcRf=HcR.Flatten();
    assert(IsHermitian(HcRf,epsHerm));

    Matrix4CT HcL=ContractHC(LW,RWm); //?
    MatrixCT  HcLf=HcL.Flatten();
    assert(IsHermitian(HcLf,epsHerm));

    Hacf=0.5*(Hacf + ~Hacf);
    HcRf=0.5*(HcRf + ~HcRf );
    HcLf=0.5*(HcLf + ~HcLf); //?
    LapackEigenSolver<dcmplx> asolver;
    auto [Mf ,eM ]=asolver.Solve(Hacf,1e-12,1);
    auto [GRf,eGR]=asolver.Solve(HcRf,1e-12,1);
    auto [GLf,eGL]=asolver.Solve(HcLf,1e-12,1); //?

    rd.eM =eM(1);
    rd.eGR=eGR(1);
    rd.eGL=eGL(1);

    assert(fabs(GRf(1,1))>1e-5);
    dcmplx phase=GRf(1,1)/fabs(GRf(1,1));
//    assert(fabs(phase)-1.0<1e-14);
    GRf*=conj(phase); //Take out arbitrary phase angle
//
    assert(fabs(GLf(1,1))>1e-5);
    phase=GLf(1,1)/fabs(GLf(1,1));
//    assert(fabs(phase)-1.0<1e-14);
    GLf*=conj(phase); //Take out arbitrary phase angle
    assert(GLf(1,1)/fabs(GLf(1,1))==1.0);
    assert(GRf(1,1)/fabs(GRf(1,1))==1.0);
//    cout << std::fixed << "G,Gm,GR,GL,eM, eR, eL" << itsG(1,1) << " " << left_neighbour->itsG(1,1) << " " << GRf(1,1) << " " << GLf(1,1) << " " << eM << " " << eGR << " " << eGL << endl;

    Tensor3 M(d,D1,D2,"M",itsSiteNumber);
    M.UnFlatten(VectorCT(Mf.GetColumn(1)));
    MatrixCT GR=UnFlatten(VectorCT(GRf.GetColumn(1)));
    MatrixCT GL=UnFlatten(VectorCT(GLf.GetColumn(1))); //?
    MatrixCT GRdagger=~GR;
    MatrixCT GLdagger=~GL; //?

    assert(fabs(1.0-std::real(Trace(GL*GLdagger)))<1e-14);
    assert(fabs(1.0-std::real(Trace(GR*GRdagger)))<1e-14);

    VectorRT sGR,sGL;
    LapackSVDSolver<dcmplx> SVDsolver;
    {
        auto [U,s,VT]=SVDsolver.SolveAll(GR,1e-14);
        rd.minsR=Min(s.GetDiagonal());
        assert(fabs(1.0-s.GetDiagonal()*s.GetDiagonal())<1e-14);
        itsRightBond->SetSingularValues(s,0.0);
        rd.dsR= itsRightBond->GetMaxDelta();
        sGR=s.GetDiagonal();
    }
    {
        auto [U,s,VT]=SVDsolver.SolveAll(GL,1e-14);
        rd.minsL=Min(s.GetDiagonal());
        assert(fabs(1.0-s.GetDiagonal()*s.GetDiagonal())<1e-14);
        itsLeft_Bond->SetSingularValues(s,0.0);
        rd.dsL= itsLeft_Bond->GetMaxDelta();
        sGL=s.GetDiagonal();
    }

    MatrixCT ML=M.Flatten(DLeft);
    MatrixCT MR=M.Flatten(DRight);
    MatrixCT MGR=ML*GRdagger;
    MatrixCT GLM=GLdagger*MR; //?
    Tensor3 Anew(d,D1,D2,"A",itsSiteNumber),Bnew(d,D1,D2,"B",itsSiteNumber);
    {
        auto [Ul,sl,VTl]=SVDsolver.SolveAll(MGR,1e-14);
        auto [Ur,sr,VTr]=SVDsolver.SolveAll(GLM,1e-14); //?
//        double s2l=sl.GetDiagonal()*sl.GetDiagonal();
//        double srl=sr.GetDiagonal()*sr.GetDiagonal();
        MatrixCT Af=Ul*VTl;
        MatrixCT Bf=Ur*VTr;
        Anew.UnFlatten(DLeft ,Af);
        Bnew.UnFlatten(DRight,Bf);
        rd.etaL=FrobeniusNorm(ML-Af*GR);
        rd.etaR=FrobeniusNorm(MR-GL*Bf); // big!!
        VectorRT sGR2=DirectMultiply(sGR,sGR); //sGR^2
        VectorRT sGL2=DirectMultiply(sGL,sGL); //sGL^2
        rd.etaRs=Max(fabs(sl.GetDiagonal()-sGR2));
        rd.etaLs=Max(fabs(sr.GetDiagonal()-sGL2));
    }

    itsM=M;
    itsA=Anew;
    itsB=Bnew;
    itsG=GR;
    left_neighbour->itsG=GL;


    for (int n=0;n<d;n++)
    {
        rd.etaG1+=FrobeniusNorm(itsA(n)*GR-GL*itsB(n));
        rd.etaG2+=FrobeniusNorm(left_neighbour->itsA(n)*GL-GR*left_neighbour->itsB(n)); //Fails
    }

    return rd;
}

RefineData::RefineData()
    : etaL     (0), etaR     (0)
    , etaLs    (0), etaRs    (0)
    , etaG1    (0), etaG2    (0)
    , dsL      (0), dsR      (0)
    , minsL    (0), minsR    (0)
{}

double iMPSSite::GetExpectation(const MatrixOR& Wcell,const UnitcellMPSType& AB) const
{
    assert(Wcell.GetForm()==RegularLower);
    assert(IsLowerTriangular(Wcell));

    auto [A,Am,B,Bp]=AB;
    VectorCT  R =A .GetTMEigenVector(DLeft );
    return A.GetExpectation(Wcell);
 }

double iMPSSite::GetExpectation(const SiteOperator* so) const
{
    assert(so);
    MatrixOR W=so->GetW();
    assert(W.GetForm()==RegularLower);
    assert(IsLowerTriangular(W));
    return itsA.GetExpectation(W);
}


void iMPSSite::SVDTransfer(Direction lr,const DiagonalMatrixRT& s, const MatrixCT& UV)
{
    assert(false);
}
void iMPSSite::SVDTransfer(Direction lr,const MatrixCT& UV)
{
    assert(false);

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
