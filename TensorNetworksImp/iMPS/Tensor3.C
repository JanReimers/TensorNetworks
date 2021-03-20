#include "TensorNetworksImp/iMPS/Tensor3.H"
#include "TensorNetworksImp/Typedefs.H"
#include "Operators/OperatorValuedMatrix.H"
#include "NumericalMethods/LapackQRSolver.H"
#include "NumericalMethods/ArpackEigenSolver.H"
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



// Rank Revealing QL/LQ
MatrixCT Tensor3::QLRR(Direction lr,double eps)
{
    LapackQRSolver <dcmplx>  solver;
    MatrixCT Mf=Flatten(lr);
    MatrixCT L,Q;

    switch (lr)
    {
        case DLeft:
            std::tie(Q,L)=solver.SolveRankRevealingQL(Mf,eps);
        break;
        case DRight:
            std::tie(L,Q)=solver.SolveRankRevealingLQ(Mf,eps);
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


} //namespace
