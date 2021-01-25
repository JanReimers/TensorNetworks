#include "Operators/SiteOperatorImp.H"
#include "TensorNetworksImp/SpinCalculator.H"
#include "oml/diagonalmatrix.h"

namespace TensorNetworks
{

//
//  Build with Dw=1 identity operators
//
SiteOperatorImp::SiteOperatorImp(int d)
    : itsd(d)
    , itsDw(1,1)
    , itsTruncationError(0.0)
    , isShapeDirty(true) //Init_lr will turn this off
    , isData_Dirty(true) //Init_lr will turn this off
    , itsWs(d,d)
{
    MatrixRT I0(1,1),I1(1,1);
    I0(1,1)=0.0;
    I1(1,1)=1.0;

    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            itsWs(m+1,n+1)= (m==n) ? I1 : I0;
            assert(itsWs(m+1,n+1).GetNumRows()==itsDw.Dw1);
            assert(itsWs(m+1,n+1).GetNumCols()==itsDw.Dw2);
        }
}

SiteOperatorImp::SiteOperatorImp(int d, double S, SpinOperator so) //Construct spin operator
    : itsd(d)
    , itsDw(1,1)
    , itsTruncationError(0.0)
    , isShapeDirty(true) //Init_lr will turn this off
    , isData_Dirty(true) //Init_lr will turn this off
    , itsWs(d,d)
{
    SpinCalculator sc(S);
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            itsWs(m+1,n+1)=sc.Get(m,n,so);
            assert(itsWs(m+1,n+1).GetNumRows()==itsDw.Dw1);
            assert(itsWs(m+1,n+1).GetNumCols()==itsDw.Dw2);
        }
}
//
//  Build from a W rep object
//
SiteOperatorImp::SiteOperatorImp(int d, const OperatorClient* H)
    : itsd(d)
    , itsDw(H->GetDw12())
    , itsTruncationError(0.0)
    , isShapeDirty(true) //Init_lr will turn this off
    , isData_Dirty(true) //Init_lr will turn this off
    , itsWs(d,d)
{
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            itsWs(m+1,n+1)=H->GetW(m,n);
            assert(itsWs(m+1,n+1).GetNumRows()==itsDw.Dw1);
            assert(itsWs(m+1,n+1).GetNumCols()==itsDw.Dw2);
        }
}

//
// Build from a trotter decomp.
//
SiteOperatorImp::SiteOperatorImp(int d, Direction lr,const MatrixRT& U, const DiagonalMatrixRT& s)
    : itsd(d)
    , itsDw()
    , itsTruncationError(0.0)
    , isShapeDirty(true) //Init_lr will turn this off
    , isData_Dirty(true) //Init_lr will turn this off
    , itsWs(d,d)
{
    int Dw=s.GetNumRows();
    if (lr==DLeft)
    {
        itsDw=Dw12(1,Dw);
        int i1=1; //Linear index for (m,n) = 1+m+p*n
        //  Fill W^(m,n)_w matrices
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,i1++)
            {
                itsWs(m+1,n+1)=MatrixRT(1,Dw);
                for (int w=1; w<=Dw; w++)
                    itsWs(m+1,n+1)(1,w)=U(i1,w)*sqrt(s(w,w));
                //cout << "Left itsWs(" << m << "," << n << ") = " << itsWs(m+1,n+1) << endl;
                assert(itsWs(m+1,n+1).GetNumRows()==itsDw.Dw1);
                assert(itsWs(m+1,n+1).GetNumCols()==itsDw.Dw2);
            }
    }
    else if (lr==DRight)
    {
        itsDw=Dw12(Dw,1);
        int i2=1; //Linear index for (m,n) = 1+m+p*n
        //  Fill W^(m,n)_w matrices
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,i2++)
            {
                itsWs(m+1,n+1)=MatrixRT(Dw,1);
                for (int w=1; w<=Dw; w++)
                    itsWs(m+1,n+1)(w,1)=sqrt(s(w,w))*U(w,i2); //U is actually VT
                //cout << "Right itsWs(" << m << "," << n << ") = " << itsWs(m+1,n+1) << endl;
                assert(itsWs(m+1,n+1).GetNumRows()==itsDw.Dw1);
                assert(itsWs(m+1,n+1).GetNumCols()==itsDw.Dw2);
            }
    }
    else
    {
        // Must be been called with one of the spin decomposition types.
        assert(false);
    }
}
//
// Construct with W operator
//
SiteOperatorImp::SiteOperatorImp(int d, const TensorT& W)
    : itsd(d)
    , itsDw()
    , itsTruncationError(0.0)
    , isShapeDirty(true) //Init_lr will turn this off
    , isData_Dirty(true) //Init_lr will turn this off
    , itsWs(W)
{
    int Dw=itsWs(1,1).GetNumRows();
    assert(itsWs(1,1).GetNumCols()==Dw);
    itsDw=Dw12(Dw,Dw);
}


SiteOperatorImp::~SiteOperatorImp()
{
    //dtor
}

void SiteOperatorImp::CheckDws() const
{
#ifdef DEBUG
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            const MatrixRT& W=GetiW(m,n);
            assert(W.GetNumRows()==itsDw.Dw1);
            assert(W.GetNumCols()==itsDw.Dw2);
        }
#endif
}

void SiteOperatorImp::SetNeighbours(SiteOperator* left, SiteOperator* right)
{
    assert(left || right); //At least one needs to be non zero
    itsLeft_Neighbour=dynamic_cast<SiteOperatorImp*>(left);
    itsRightNeighbour=dynamic_cast<SiteOperatorImp*>(right);
    assert(!left  || itsLeft_Neighbour); //if left is nonzero then did the cast work?
    assert(!right || itsRightNeighbour);
}

void SiteOperatorImp::SetiW(int m, int n, const MatrixRT& W)
{
    assert(W.GetNumRows()==itsDw.Dw1);
    assert(W.GetNumCols()==itsDw.Dw2);
    if (itsWs(m+1,n+1).GetLimits()!=W.GetLimits())
        isShapeDirty=true;
    itsWs(m+1,n+1)=W;
    isData_Dirty=true;
}

void SiteOperatorImp::SetLimits()
{
    SetLimits(itsDw,itsWs);
}

void SiteOperatorImp::SetLimits(Dw12& Dw,TensorT& Ws)
{
    int d=Ws.GetNumRows();
    Dw.w1_first.SetLimits(Dw.Dw2);
    Dw.w2_last .SetLimits(Dw.Dw1);
//    Fill(Dw.w1_first,1);
//    Fill(DW.w2_last ,Dw.Dw2);
    Fill(Dw.w1_first,Dw.Dw1);
    Fill(Dw.w2_last ,1);
    for (int m=0; m<d; m++)
        for (int n=0; n<d; n++)
        {
            const MatrixRT& W=Ws(m+1,n+1);
            for (int w1=1; w1<=Dw.Dw1; w1++)
                for (int w2=1; w2<=Dw.Dw2; w2++)
                    if (W(w1,w2)!=0.0)
                    {
                        if (Dw.w1_first(w2)>w1) Dw.w1_first(w2)=w1;
                        if (Dw.w2_last (w1)<w2) Dw.w2_last (w1)=w2;
                    }
//            cout << "W(" << m << "," << n << ")=" << W << endl;
//            cout << "w1_first=" << itsDw12.w1_first << endl;
//            cout << "w2_last =" << itsDw12.w2_last  << endl;
        }

}

void  SiteOperatorImp::AccumulateTruncationError(double err)
{
    itsTruncationError=sqrt(itsTruncationError*itsTruncationError+err*err);
}


void SiteOperatorImp::Combine(const SiteOperator* O2,double factor)
{
    const SiteOperatorImp* O2i(dynamic_cast<const SiteOperatorImp*>(O2));
    assert(O2i);
    Dw12 O2Dw=O2i->itsDw;
    Dw12 Dw(itsDw.Dw1*O2Dw.Dw1,itsDw.Dw2*O2Dw.Dw2);

//    cout << "MPO D1,D2=" << itsDw12.Dw1 << " " << itsDw12.Dw2 << " ";
//    cout << "O2  D1,D2=" << O2Dw.Dw1 << " " << O2Dw.Dw2 << " ";
//    cout << "New D1,D2=" << Dw.Dw1 << " " << Dw.Dw2 << endl;


    TensorT newWs(itsd,itsd);
    for (int m=0; m<itsd; m++)
        for (int o=0; o<itsd; o++)
        {
            MatrixRT Wmo(Dw.Dw1,Dw.Dw2);
            Fill(Wmo,0.0);
            for (int n=0; n<itsd; n++)
            {
                const MatrixRT& W1=GetiW(m,n);
                const MatrixRT& W2=O2i->GetiW(n,o);
                int w1=1;
                for (int w11=1; w11<=itsDw.Dw1; w11++)
                    for (int w12=1; w12<=   O2Dw.Dw1; w12++,w1++)
                    {
                        int w2=1;
                        for (int w21=1; w21<=itsDw.Dw2; w21++)
                            for (int w22=1; w22<=   O2Dw.Dw2; w22++,w2++)
                                Wmo(w1,w2)+=W1(w11,w21)*W2(w12,w22);
                    }
            }
            newWs(m+1,o+1)=factor*Wmo;
        }
    itsWs=newWs;

    itsDw=Dw;
    isShapeDirty=true;
    isData_Dirty=true;
    Init_lr();
}


void SiteOperatorImp::Report(std::ostream& os) const
{
    Dw12 Dw=GetDw12();
    os
    << std::setw(3) << Dw.Dw1 << " "
    << std::setw(3) << Dw.Dw2 << " "
    << std::setw(3) << itsDw.Dw1 << " "
    << std::setw(3) << itsDw.Dw2 << " "
    << std::scientific << std::setprecision(1) << itsTruncationError
    << std::fixed << " " << GetFrobeniusNorm()
    << std::fixed << " " << GetiFrobeniusNorm()
    << " " << GetNormStatus(1e-13)
//    << " " << itsDw12.w1_first << " " << itsDw12.w2_last
    ;
}

double SiteOperatorImp::GetFrobeniusNorm() const
{
    double fn=0.0;
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
            fn+=FrobeniusNorm(GetW(n,m));
    return fn;
}

double SiteOperatorImp::GetiFrobeniusNorm() const
{
    double fn=0.0;
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
            fn+=FrobeniusNorm(GetiW(n,m));
    return fn;
}

bool SiteOperatorImp::isOrthonormal(Direction lr,const MatrixRT& Q) const
{
    bool ret=false;
    double d=itsd;
    switch (lr)
    {
    case DLeft:
        ret=IsUnit(Transpose(Q)*Q/d,1e-13);
        break;
    case DRight:
        ret=IsUnit(Q*Transpose(Q)/d,1e-13);
        break;
    default:
        assert(false);
    }
    return ret;
}

char SiteOperatorImp::GetNormStatus(double eps) const
{
    char ret='W'; //Not normalized
    {
        MatrixRT QL=Reshape(DLeft,1);
        if (QL.GetNumRows()==0)
            ret='l';
        else if (isOrthonormal(DLeft,QL))
            ret='L';
    }
    if (ret!='l')
    {
        MatrixRT QR=Reshape(DRight,1);
        if (QR.GetNumCols()==0)
            ret='r';
        else if (isOrthonormal(DRight,QR))
        {
            if (ret=='L')
                ret='I';
            else
                ret='R';
        }
    }
    return ret;
}


} //namespace
//---------------------------------------------------------------------------------
//
//  Make template instance
//
#define TYPE Matrix<double>
#include "oml/src/matrix.cpp"
#undef TYPE
#define TYPE int
#include "oml/src/vector.cpp"
