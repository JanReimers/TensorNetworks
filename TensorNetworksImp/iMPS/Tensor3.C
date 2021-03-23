#include "TensorNetworksImp/iMPS/Tensor3.H"
#include "TensorNetworksImp/Typedefs.H"
#include "Operators/OperatorValuedMatrix.H"
#include "NumericalMethods/LapackQRSolver.H"
#include "NumericalMethods/LapackEigenSolver.H"
#include "NumericalMethods/ArpackEigenSolver.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include "NumericalMethods/LapackLinearSolver.H"
#include "Containers/Matrix6.H"
#include "oml/diagonalmatrix.h"
#include "oml/random.h"

using std::cout;
using std::endl;

namespace TensorNetworks
{

Tensor3::Tensor3(int d, int D1, int D2, int mnlow)
: itsmnLow(mnlow)
{
    for (int n=0; n<d; n++)
    {
        itsMs.push_back(MatrixCT(D1,D2));
        Fill(itsMs.back(),std::complex<double>(0.0));
    }
}


void Tensor3::InitializeWith(State state,int sgn)
{
    int d=Getd();
    switch (state)
    {
    case Product :
    {
        int i=1;
        for (auto& M:itsMs)
            if (i<=GetD2())
                M(1,i++)=std::complex<double>(sgn); //Left normalized
        break;
    }
    case Random :
    {
        for (auto& M:itsMs)
        {
            FillRandom(M);
            M*=1.0/sqrt(d*GetD1()*GetD2()); //Try and keep <psi|psi>~O(1)
        }
        break;
    }
    case Neel :
    {
        for (auto& M:itsMs)
            Fill(M,dcmplx(0.0));
        if (sgn== 1)
            itsMs[0  ](1,1)=1.0;
        if (sgn==-1)
            itsMs[d-1](1,1)=1.0;

        break;
    }
    }
}

void Tensor3::Unit(int n)
{
    int n1=n-itsmnLow;
    int d=Getd();
    assert(n1<d);
    assert(n1>=0);
    ::Unit(itsMs[n1]);
}

void Tensor3::UnFlatten(const VectorCT& v)
{
    auto [d,D1,D2]=GetDimensions();
    assert(v.size()==d*D1*D2);
    int nij=1;
    for (index_t j=1;j<=D2;j++)
        for (index_t i=1;i<=D1;i++)
             for (auto& M:itsMs)
                M(i,j)=v(nij++);
}


MatLimits Tensor3::GetLimits() const
{
    assert(itsMs.size()>0);
    MatLimits l=itsMs[0].GetLimits();
#ifdef DEBUG
    for (auto& M:itsMs) assert(M.GetLimits()==l);
#endif // DEBUG
    return l;
}

bool Tensor3::IsSquare () const
{
    assert(itsMs.size()>0);
    bool sq=itsMs[0].IsSquare();
#ifdef DEBUG
    for (auto& M:itsMs) assert(M.IsSquare()==sq);
#endif // DEBUG
    return sq;

}

const MatrixCT& Tensor3::operator()(int n) const
{
    int n1=n-itsmnLow;
    int d=Getd();
    assert(n1<d);
    assert(n1>=0);
    return itsMs[n1];
}
MatrixCT& Tensor3::operator()(int n)
{
    int n1=n-itsmnLow;
    int d=Getd();
    assert(n1<d);
    assert(n1>=0);
    return itsMs[n1];
}

MatrixCT Tensor3::InitNorm(Direction lr) const
{
    int D= (lr==DLeft) ? GetD2() : GetD1();
    MatrixCT N(D,D);
    Fill(N,std::complex<double>(0.0));
    return N;
}


MatrixCT  Tensor3::GetNorm(Direction lr) const
{
    MatrixCT ret=InitNorm(lr);
    switch(lr)
    {
    case DLeft:
        for (const MatrixCT& M:itsMs)
            ret+=conj(Transpose((M)))*(M);
        break;
    case DRight:
        for (const MatrixCT& M:itsMs)
            ret+=M*conj(Transpose(M));
        break;
    }
    return ret;
}

MatrixCT  Tensor3::GetNorm(Direction lr, const DiagonalMatrixRT& lambda) const
{
    MatrixCT ret=InitNorm(lr);
    switch(lr)
    {
    case DLeft:
        for (const MatrixCT& M:itsMs)
            ret+=conj(Transpose(M))*lambda*lambda*M;
        break;
    case DRight:
        for (const MatrixCT& M:itsMs)
            ret+=M*lambda*lambda*conj(Transpose(M));
        break;
    }
    return ret;

}

MatrixCT  Tensor3::Flatten(Direction lr) const
{
    auto [d,D1,D2]=GetDimensions();
    int k=1; // compound {n,i} index
    MatrixCT F;
    switch (lr)
    {
    case DLeft:
        F.SetLimits(d*D1,D2);
        for (auto& M:itsMs)
            for (int i1=1; i1<=D1; i1++,k++)
                for (int i2=1; i2<=D2; i2++)
                    F(k,i2)=M(i1,i2);
        break;
    case DRight:
        F.SetLimits(D1,d*D2);
        for (auto& M:itsMs)
            for (int i2=1; i2<=D2; i2++,k++)
                for (int i1=1; i1<=D1; i1++)
                    F(i1,k)=M(i1,i2);
        break;
    }
    return F;
}


void Tensor3::UnFlatten(Direction lr,const MatrixCT& F)
{
    auto [d,D1,D2]=GetDimensions();
    int k=1; // compound {n,i} index
    switch (lr)
    {
    case DLeft:
        assert(F.GetNumCols()==D2);
        assert(F.GetNumRows()==d*D1);
        for (auto& M:itsMs)
            for (int i1=1; i1<=D1; i1++,k++)
                for (int i2=1; i2<=D2; i2++)
                    M(i1,i2)=F(k,i2);
        break;
    case DRight:
        assert(F.GetNumCols()==d*D2);
        assert(F.GetNumRows()==D1);
        for (auto& M:itsMs)
            for (int i2=1; i2<=D2; i2++,k++)
                for (int i1=1; i1<=D1; i1++)
                    M(i1,i2)=F(i1,k);
        break;
    }
}

void Tensor3::Multiply(Direction lr,const MatrixCT& G)
{
    switch (lr)
    {
    case DLeft:
        for (auto& M:itsMs)
            M=G*M;
        break;
    case DRight:
        for (auto& M:itsMs)
            M=M*G;
        break;
    }
}


// Rank Revealing QL/LQ
MatrixCT Tensor3::QLRR(Direction lr,double eps)
{
    LapackQRSolver <dcmplx>  solver;
    MatrixCT Mf=Flatten(lr);
    MatrixCT L,Q;

    double neps=3*eps;
    switch (lr)
    {
        case DLeft:
            std::tie(Q,L)=solver.SolveRankRevealingQL(Mf,eps);
            assert(Max(fabs(Mf-Q*L))<neps);
            for (index_t i:L.rows())
            {
                dcmplx phase=L(i,i)/fabs(L(i,i));
                L.GetRow(i)*=conj(phase);
                Q.GetColumn(i)*=phase;
            }
            assert(Max(fabs(Mf-Q*L))<neps);
        break;
        case DRight:
            std::tie(L,Q)=solver.SolveRankRevealingLQ(Mf,eps);
            {
                double err=Max(fabs(Mf-L*Q));
                if (err>=neps) cout << "Max(fabs(Mf-L*Q))=" << err << endl;
                assert(err<neps);

            }
            for (index_t i:L.cols())
            {
                dcmplx phase=L(i,i)/fabs(L(i,i));
                L.GetColumn(i)*=conj(phase);
                Q.GetRow(i)*=phase;
            }
            assert(Max(fabs(Mf-L*Q))<neps);
        break;
    }
    assert(!isnan(L));
    assert(!isinf(L));
    assert(IsLowerTriangular(L));

    UnFlatten(lr,Q);
    return L;
}

MatrixCT Tensor3::GetTransferMatrix(Direction lr) const
{
    auto [d,D1,D2]=GetDimensions();
    Matrix4CT T(D1,D1,D2,D2);
    T.Fill(dcmplx(0.0));

    for (index_t i1=1;i1<=D1;i1++)
        for (index_t i2=1;i2<=D2;i2++)
            for (index_t j1=1;j1<=D1;j1++)
                for (index_t j2=1;j2<=D2;j2++)
                {
                    dcmplx t(0.0);
                    switch (lr)
                    {
                    case DLeft:
                        for (auto& M:itsMs)
                            t+=conj(M(i1,i2))*M(j1,j2);
                        break;
                    case DRight:
                        for (auto& M:itsMs)
                            t+=M(i1,i2)*conj(M(j1,j2));
                        break;
                    }
                    T(i1,j1,i2,j2)=t;
                }
    return T.Flatten();
}

Matrix6CT Tensor3::GetTransferMatrix(Direction lr,const MatrixOR& W) const
{
    auto [d,D1,D2]=GetDimensions();
    auto [X1,X2]=W.GetChi12();
    assert(d==W.Getd());
    Matrix6CT Tw(X1+2,D1,D1,X2+2,D2,D2,1);
    OperatorZ Z(d);
    Tw.Fill(dcmplx(0.0));

    for (index_t w1=0;w1<=X1+1;w1++)
    for (index_t w2=0;w2<=X2+1;w2++)
    {
        if (!(W(w1,w2)==Z))
        {
        for (index_t i1=1;i1<=D1;i1++)
        for (index_t i2=1;i2<=D2;i2++)
            for (index_t j1=1;j1<=D1;j1++)
            for (index_t j2=1;j2<=D2;j2++)
                {
                    dcmplx t(0.0);
                    switch (lr)
                    {
                    case DLeft:
                        for (int m=0;m<d;m++)
                        for (int n=0;n<d;n++)
                            t+=W(w1,w2)(m,n)*conj(itsMs[m](i1,i2))*itsMs[n](j1,j2);
                        break;
                    case DRight:
                        for (int m=0;m<d;m++)
                        for (int n=0;n<d;n++)
                            t+=W(w1,w2)(m,n)*itsMs[m](i1,i2)*conj(itsMs[n](j1,j2));
                        break;
                    }
                    Tw(w1+1,i1,j1,w2+1,i2,j2)=t;
                }

        }
    }
    return Tw;
}

VectorCT Tensor3::GetTMEigenVector (Direction lr) const
{
    assert(GetD1()>1);
    assert(GetD2()>1); //Arpack eigen solver doesn't like 1x1 matricies.
    MatrixCT T=GetTransferMatrix(lr);
    EigenSolver<dcmplx>* solver=new ArpackEigenSolver<dcmplx>();
    MatrixCT U;
    VectorCT e;
    switch (lr)
    {
    case DLeft:
        std::tie(U,e)=solver->SolveRightNonSym(T,1e-13,1);
        break;
    case DRight:
        std::tie(U,e)=solver->SolveLeft_NonSym(T,1e-13,1);
        break;
    }
    delete solver;
    double   eigenValue =real(e(1));
    VectorCT V=U.GetColumn(1);
    assert(fabs(eigenValue-1.0)<1e-13);
    dcmplx phase=V(1)/fabs(V(1));
    assert(fabs(phase)-1.0<1e-14);
    V*=conj(phase); //Take out arbitrary phase angle

    return V;
}

Tensor3::LRWType Tensor3::GetLW(const MatrixOR& W) const
{
    IsUnit(GetNorm(DLeft),1e-13);
    auto [d,D1,D2]=GetDimensions();
    auto [X1,X2]=W.GetChi12();
    assert(X1==X2);
    int Dw=X1+2;

    Matrix6CT TWL=GetTransferMatrix(DLeft ,W);
    MatrixCT  TL =GetTransferMatrix(DLeft );
    VectorCT  R  =GetTMEigenVector(DLeft );
    assert(Max(fabs(TL-TWL.SubMatrix(1 ,1 ).Flatten()))<1e-15);
    assert(Max(fabs(TL-TWL.SubMatrix(Dw,Dw).Flatten()))<1e-15);
    Tensor3 LW(Dw,D1,D2,1);
    LW.Unit(Dw);
    cout << std::setprecision(3);
    for (int w=Dw-1;w>1;w--)
    {
        assert(Max(fabs(TWL.SubMatrix(w,w).Flatten()))==0.0);
        for (int w1=w+1;w1<=Dw;w1++)
            LW(w)+=LW(w1)*TWL.SubMatrix(w1,w);
        assert(IsHermitian(LW(w),1e-15));
    }
    //
    //  Now we need YL_1 and YR_Dw
    //
    MatrixCT YL(D1,D2),YR(D1,D2);
    Fill(YL,dcmplx(0.0));
    for (int w1=2;w1<=Dw;w1++)
        YL+=LW(w1)*TWL.SubMatrix(w1,1);

    assert(IsHermitian(YL,1e-15));
    VectorCT YLf=TensorNetworks::Flatten(YL);

    MatrixCT I(TL.GetLimits()),IL(D1,D1);
    ::Unit(I);::Unit(IL);
    VectorCT ILf=TensorNetworks::Flatten(IL);
    MatrixCT PL=OuterProduct(R,ILf);
    MatrixCT XL=I-TL+PL;
    VectorCT vL=YLf-YLf*PL;
    LapackSVDSolver<dcmplx> SVDsolver;
    {
        auto [U,s,VT]=SVDsolver.SolveAll(XL,1e-14);
        //cout << std::scientific << "XL min s=" << Min(s.GetDiagonal()) << endl;
        assert(Min(s.GetDiagonal())>1e-14);
    }

    LapackLinearSolver<dcmplx> solver;
    VectorCT LW1f=solver.Solve(vL,XL);
    LW(1 )=TensorNetworks::UnFlatten(LW1f);
    for (int w=1;w<=Dw;w++)
    {
        //cout << "LW(" << w << ")=" << LW(w) << endl;
//        cout << std::scientific << Max(fabs(LW(w)-~LW(w))) << endl;
        LW(w)=0.5*(LW(w)+~LW(w));
        assert(IsHermitian(LW(w),1e-13));
    }
    //
    //  Site energy e = (YL_1|R) = (L|YR_Dw)
    //
    dcmplx el=YLf*R;
    assert(fabs(std::imag(el))<1e-11);

    return std::make_tuple(std::real(el),LW);
}

Tensor3::LRWType Tensor3::GetRW(const MatrixOR& W) const
{
    IsUnit(GetNorm(DRight),1e-13);
    auto [d,D1,D2]=GetDimensions();
    auto [X1,X2]=W.GetChi12();
    assert(X1==X2);
    int Dw=X1+2;

    Matrix6CT TWR=GetTransferMatrix(DRight,W);
    MatrixCT  TR =GetTransferMatrix(DRight);
    VectorCT  L  =GetTMEigenVector(DRight);
    assert(Max(fabs(TR-TWR.SubMatrix(1 ,1 ).Flatten()))<1e-15);
    assert(Max(fabs(TR-TWR.SubMatrix(Dw,Dw).Flatten()))<1e-15);
    Tensor3 RW(Dw,D1,D2,1);
    RW.Unit(1);
    cout << std::setprecision(3);
    for (int w=Dw-1;w>1;w--)
    {
        assert(Max(fabs(TWR.SubMatrix(w,w).Flatten()))==0.0);
        for (int w2=1;w2<w;w2++)
            RW(w)+=TWR.SubMatrix(w,w2)*RW(w2);
        assert(IsHermitian(RW(w),1e-15));
    }
    //
    //  Now we need YL_1 and YR_Dw
    //
    MatrixCT YR(D1,D2);
    Fill(YR,dcmplx(0.0));
    for (int w2=1;w2< Dw;w2++)
        YR+=TWR.SubMatrix(Dw,w2)*RW(w2);

    assert(IsHermitian(YR,1e-15));
    VectorCT YRf=TensorNetworks::Flatten(YR);

    MatrixCT I(TR.GetLimits()),IR(D2,D2);
    ::Unit(I);::Unit(IR);
    VectorCT IRf=TensorNetworks::Flatten(IR);

    MatrixCT PR=OuterProduct(IRf,L);
    MatrixCT XR=I-TR+PR;
    VectorCT vR=YRf-PR*YRf;

    LapackSVDSolver<dcmplx> SVDsolver;
    {
        auto [U,s,VT]=SVDsolver.SolveAll(XR,1e-14);
        //cout << std::scientific << "XR min s=" << Min(s.GetDiagonal()) << endl;
        assert(Min(s.GetDiagonal())>1e-14);
    }

    LapackLinearSolver<dcmplx> solver;
    VectorCT RWDf=solver.Solve(XR,vR);
    RW(Dw)=TensorNetworks::UnFlatten(RWDf);
    for (int w=1;w<=Dw;w++)
    {
        RW(w)=0.5*(RW(w)+~RW(w));
        assert(IsHermitian(RW(w),1e-12));
    }
    //
    //  Site energy e = (YL_1|R) = (L|YR_Dw)
    //
    dcmplx er=L*YRf;
    assert(fabs(std::imag(er))<1e-12);

    return std::make_tuple(std::real(er),RW);
}

double Tensor3::GetExpectation(const MatrixOR& W) const
{
    IsUnit(GetNorm(DLeft),1e-13);
    auto [d,D1,D2]=GetDimensions();
    auto [X1,X2]=W.GetChi12();
    assert(X1==X2);
    int Dw=X1+2;

    Matrix6CT TWL=GetTransferMatrix(DLeft ,W);
    MatrixCT  TL =GetTransferMatrix(DLeft );
    VectorCT  R  =GetTMEigenVector(DLeft );
    Tensor3 LW(Dw,D1,D2,1);
    LW.Unit(Dw);
    for (int w=Dw-1;w>1;w--)
    {
        assert(Max(fabs(TWL.SubMatrix(w,w).Flatten()))==0.0);
        for (int w1=w+1;w1<=Dw;w1++)
            LW(w)+=LW(w1)*TWL.SubMatrix(w1,w);
        assert(IsHermitian(LW(w),1e-15));
    }
    //
    //  Now we need YL_1 and YR_Dw
    //
    MatrixCT YL(D1,D2);
    Fill(YL,dcmplx(0.0));
    for (int w1=2;w1<=Dw;w1++)
        YL+=LW(w1)*TWL.SubMatrix(w1,1);

    assert(IsHermitian(YL,1e-15));
    VectorCT YLf=TensorNetworks::Flatten(YL);
    //
    //  Site energy e = (YL_1|R) = (L|YR_Dw)
    //
    dcmplx el=YLf*R;
    assert(fabs(std::imag(el))<1e-12);

    return std::real(el);
}

} //namespace
