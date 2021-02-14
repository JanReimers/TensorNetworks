#include "Operators/SiteOperatorImp.H"
#include "Operators/OperatorClient.H"
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
    , itsWs(d,d)
    , itsWOvM(d)
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
    : SiteOperatorImp(d)
{
    SpinCalculator sc(S);
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            itsWs(m+1,n+1)=sc.Get(m,n,so);
            assert(itsWs(m+1,n+1).GetNumRows()==itsDw.Dw1);
            assert(itsWs(m+1,n+1).GetNumCols()==itsDw.Dw2);
        }
    SyncWtoO();
}
//
//  Build from a W rep object
//
SiteOperatorImp::SiteOperatorImp(int d, const OperatorClient* H)
    : SiteOperatorImp(d)
{
    itsDw=H->GetDw12();
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            itsWs(m+1,n+1)=H->GetW(m,n);
            assert(itsWs(m+1,n+1).GetNumRows()==itsDw.Dw1);
            assert(itsWs(m+1,n+1).GetNumCols()==itsDw.Dw2);
        }
    SyncWtoO();
}

//
// Build from a trotter decomp.
//
SiteOperatorImp::SiteOperatorImp(int d, Direction lr,const MatrixRT& U, const DiagonalMatrixRT& s)
    : SiteOperatorImp(d)
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
    SyncWtoO();
}
//
// Construct with W operator
//
SiteOperatorImp::SiteOperatorImp(int d, const TensorT& W)
    : SiteOperatorImp(d)
{
    itsWs=W;
    int Dw=itsWs(1,1).GetNumRows();
    assert(itsWs(1,1).GetNumCols()==Dw);
    itsDw=Dw12(Dw,Dw);
    SyncWtoO();
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
            const MatrixRT& W=GetW(m,n);
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

void SiteOperatorImp::SyncWtoO()
{
    itsWOvM.SetChi12(itsDw.Dw1-2,itsDw.Dw2-2);
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            const MatrixRT& W=GetW(m,n);
            for (index_t i:itsWOvM.rows())
            for (index_t j:itsWOvM.cols())
                itsWOvM(i,j)(m,n)=W(i+1,j+1);
        }
}

void SiteOperatorImp::SyncOtoW()
{
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            MatrixRT W(itsDw.Dw1,itsDw.Dw2);
            for (index_t i:itsWOvM.rows())
            for (index_t j:itsWOvM.cols())
                W(i+1,j+1)=itsWOvM(i,j)(m,n);
            SetW(m,n,W);
        }

}

void SiteOperatorImp::SetW(int m, int n, const MatrixRT& W)
{
    assert(W.GetNumRows()==itsDw.Dw1);
    assert(W.GetNumCols()==itsDw.Dw2);
    itsWs(m+1,n+1)=W;

//    if (W.GetNumRows()!=itsWOvM.GetNumRows() || W.GetNumCols()==itsWOvM.GetNumCols())
    itsWOvM.SetChi12(itsDw.Dw1-2,itsDw.Dw2-2);
    for (index_t i:itsWOvM.rows())
    for (index_t j:itsWOvM.cols())
          itsWOvM(i,j)(m,n)=W(i+1,j+1);
}

void SiteOperatorImp::SetLimits()
{
    itsDw.w1_first.SetLimits(itsDw.Dw2);
    itsDw.w2_last .SetLimits(itsDw.Dw1);
//    Fill(Dw.w1_first,1);
//    Fill(DW.w2_last ,Dw.Dw2);
    Fill(itsDw.w1_first,itsDw.Dw1);
    Fill(itsDw.w2_last ,1);
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            const MatrixRT& W=itsWs(m+1,n+1);
            for (int w1=1; w1<=itsDw.Dw1; w1++)
                for (int w2=1; w2<=itsDw.Dw2; w2++)
                    if (W(w1,w2)!=0.0)
                    {
                        if (itsDw.w1_first(w2)>w1) itsDw.w1_first(w2)=w1;
                        if (itsDw.w2_last (w1)<w2) itsDw.w2_last (w1)=w2;
                    }
//            cout << "W(" << m << "," << n << ")=" << W << endl;
//            cout << "w1_first=" << Dw.w1_first << endl;
//            cout << "w2_last =" << Dw.w2_last  << endl;
        }

}

void  SiteOperatorImp::AccumulateTruncationError(double err)
{
    itsTruncationError=sqrt(itsTruncationError*itsTruncationError+err*err);
}


void SiteOperatorImp::Product(const SiteOperator* O2)
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
                Wmo+=TensorProduct(GetW(m,n),O2i->GetW(n,o));
            newWs(m+1,o+1)=Wmo;
        }
    itsWs=newWs;  //Use SetiW instead

    itsDw=Dw;
}


void SiteOperatorImp::Report(std::ostream& os) const
{
    os
    << std::setw(3) << itsDw.Dw1 << " "
    << std::setw(3) << itsDw.Dw2 << "   "
    << std::scientific << std::setprecision(1) << itsTruncationError
    << " " << std::fixed << std::setprecision(1) << std::setw(5) << GetFrobeniusNorm()
    << " " << std::setw(4) << GetNormStatus(1e-13)
    << " " << std::setw(4) << GetUpperLower(1e-13)
    << " " << std::setw(4) << GetLRB() //Left, Bulk, Right
//    << " " << itsDw12.w1_first << " " << itsDw12.w2_last
    ;
}

char SiteOperatorImp::GetUpperLower(double eps) const
{
    char ret=' ';
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            if (IsUpperTriangular(GetW(n,m),eps))
            {
                if (ret==' ')
                    ret='U';
                else if (ret=='L')
                    ret='M'; //Mix
            }
            else if (IsLowerTriangular(GetW(n,m),eps))
            {
                if (ret==' ')
                    ret='L';
                else if (ret=='U')
                    ret='M'; //Mix
            }
            else
                ret='F'; // Full
        }
    return ret;
}

double SiteOperatorImp::GetFrobeniusNorm() const
{
    double fn=0.0;
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
            fn+=FrobeniusNorm(GetW(n,m));
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
//    cout << "Dw1*Dw2=" << itsDw.Dw1*itsDw.Dw2 << endl;
    if (itsDw.Dw1*itsDw.Dw2>4096) return '?';
    char ret='W'; //Not normalized
    {
        MatrixRT QL=ReshapeV(DLeft);
        if (QL.GetNumRows()==0)
            ret='l';
        else if (isOrthonormal(DLeft,QL))
            ret='L';
    }
    if (ret!='l')
    {
        MatrixRT QR=ReshapeV(DRight);
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

