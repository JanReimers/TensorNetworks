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
    , itsWOvM(1,1,d,Lower)
{
    Unit(itsWOvM);
    SyncOtoW();
}

//
//  Not covered by MPO tests, try the Expectation tests.
//
SiteOperatorImp::SiteOperatorImp(int d, double S, SpinOperator so) //Construct spin operator
    : SiteOperatorImp(d)
{
    itsWOvM(0,0)=OperatorElement<double>::Create(so,S);
    SyncOtoW();
}
//
//  Build from a W rep object
//
SiteOperatorImp::SiteOperatorImp(int d, const OperatorClient* H)
    : SiteOperatorImp(d)
{
    itsWOvM=H->GetMatrixO(Lower);
    itsDw=Dw12(itsWOvM.GetNumRows(),itsWOvM.GetNumCols());
    SyncOtoW();
    CheckSync();
}

//
// Build from a trotter decomp.
//
SiteOperatorImp::SiteOperatorImp(int d, Direction lr,const MatrixRT& U, const DiagonalMatrixRT& s)
    : SiteOperatorImp(d)
{
    int Dw=s.GetNumRows();
    assert(Dw==d*d);
    assert(U.GetNumCols()==Dw);
    assert(U.GetNumRows()==Dw);

    if (lr==DLeft)
    {
        itsWOvM.SetChi12(-1,Dw-2,false);
        itsDw=Dw12(1,Dw);
        int i1=1; //Linear index for (m,n) = 1+m+p*n
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,i1++)
                for (int w=1; w<=Dw; w++)
                    itsWOvM(0,w-1)(m,n)=U(i1,w)*sqrt(s(w,w));
    }
    else if (lr==DRight)
    {
        itsWOvM.SetChi12(Dw-2,-1,false);
        itsDw=Dw12(Dw,1);
        int i2=1; //Linear index for (m,n) = 1+m+p*n
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,i2++)
                for (int w=1; w<=Dw; w++)
                    itsWOvM(w-1,0)(m,n)=sqrt(s(w,w))*U(w,i2); //U is actually VT
    }
    else
    {
        // Must have been called with one of the spin decomposition types.
        assert(false);
    }
    SyncOtoW();
}

////
// Construct with W operator. Called by iMPOImp::MakeUnitcelliMPO
//
SiteOperatorImp::SiteOperatorImp(const MatrixOR& W)
    : SiteOperatorImp(W.Getd())
{
    itsWOvM=W;
    itsDw=Dw12(itsWOvM.GetNumRows(),itsWOvM.GetNumCols());
    SyncOtoW();
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
    itsWOvM.SetChi12(itsDw.Dw1-2,itsDw.Dw2-2,false);
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            const MatrixRT& W=GetW(m,n);
            for (index_t w1:itsWOvM.rows())
            for (index_t w2:itsWOvM.cols())
                itsWOvM(w1,w2)(m,n)=W(w1+1,w2+1);
        }
}

void SiteOperatorImp::SyncOtoW()
{
    auto [X1,X2]=itsWOvM.GetChi12();
    index_t D1=X1+2;
    index_t D2=X2+2;
    if (itsDw.Dw1!=D1 || itsDw.Dw2!=D2)
    {
        for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            VectorRT lastRow=itsWs(m+1,n+1).GetRow(itsDw.Dw1);
            itsWs(m+1,n+1).SetLimits(D1,D2,true);
            if (D2<=itsDw.Dw2)
                itsWs(m+1,n+1).GetRow(D1)=lastRow.SubVector(D2);

        }
        itsDw.Dw1=D1;
        itsDw.Dw2=D2;
    }
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            MatrixRT W(itsDw.Dw1,itsDw.Dw2);
            for (index_t w1:itsWOvM.rows())
            for (index_t w2:itsWOvM.cols())
                W(w1+1,w2+1)=itsWOvM(w1,w2)(m,n);
            itsWs(m+1,n+1)=W; //Dont' call SetW here !!!
        }
    SetLimits();
}

void SiteOperatorImp::CheckSync()
{
    for (int m=0; m<itsd; m++)
    for (int n=0; n<itsd; n++)
    {
        const MatrixRT& W=GetW(m,n);
        for (index_t w1:itsWOvM.rows())
        for (index_t w2:itsWOvM.cols())
            if(itsWOvM(w1,w2)(m,n)!=W(w1+1,w2+1))
            {
                cout << "Ovw(" << w1 << "," << w2 << ")(" << m << "," << n << "),W = " << itsWOvM(w1,w2)(m,n) << " " << W(w1+1,w2+1) << endl;
            }
    }
}

SiteOperatorImp* SiteOperatorImp::GetNeighbour(Direction lr) const
{
    SiteOperatorImp* ret=0;
    switch(lr)
    {
    case DLeft:
        ret=itsRightNeighbour;
        break;
    case DRight:
        ret=itsLeft_Neighbour;
        break;
    default:
        assert(false);
    }
    assert(ret);
    return ret;
}

void SiteOperatorImp::SetLimits()
{
    itsDw.w1_first.SetLimits(itsDw.Dw2);
    itsDw.w2_last .SetLimits(itsDw.Dw1);
//    Fill(Dw.w1_first,1);
//    Fill(DW.w2_last ,Dw.Dw2);

    Fill(itsDw.w1_first,itsDw.Dw1);
    Fill(itsDw.w2_last ,1);
    for (index_t w1:itsWOvM.rows())
        for (index_t w2:itsWOvM.cols())
            if (fabs(itsWOvM(w1,w2))>0.0) //TOT should be using and eps~1e-15 here.
            {
                if (itsDw.w1_first(w2+1)>w1+1) itsDw.w1_first(w2+1)=w1+1;
                if (itsDw.w2_last (w1+1)<w2+1) itsDw.w2_last (w1+1)=w2+1;
            }

    CheckDws();
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

//    cout << "itsWOvM=" << itsWOvM.GetLimits() << " " << itsWOvM.GetUpperLower() << endl;
//    cout << "O2i->itsWOvM=" << O2i->itsWOvM.GetLimits()<< " " <<  O2i->itsWOvM.GetUpperLower()  << endl;
    TriType ul=itsWOvM.GetUpperLower();
    MatrixOR WW=TensorProduct(itsWOvM,O2i->itsWOvM);
    itsWOvM=WW;
    itsWOvM.SetUpperLower(ul);
    itsDw=Dw;
    SyncOtoW();
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
    if (IsUpperTriangular(itsWOvM,eps))
    {
        if (ret==' ')
            ret='U';
        else if (ret=='L')
            ret='M'; //Mix
    }
    else if (IsLowerTriangular(itsWOvM,eps))
    {
        if (ret==' ')
            ret='L';
        else if (ret=='U')
            ret='M'; //Mix
    }
    else
        ret='F'; // Full
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

char SiteOperatorImp::GetNormStatus(double eps) const
{
//    cout << "Dw1*Dw2=" << itsDw.Dw1*itsDw.Dw2 << endl;
    if (itsDw.Dw1*itsDw.Dw2>4096) return '?';
    char ret='W'; //Not normalized
    {
        MatrixOR V=itsWOvM.GetV(DLeft);
        if (V.GetNumRows()==0)
            ret='l';
        else if (V.IsOrthonormal(DLeft,eps))
            ret='L';
    }
    if (ret!='l')
    {
        MatrixOR V=itsWOvM.GetV(DRight);
        if (V.GetNumCols()==0)
            ret='r';
        else if (V.IsOrthonormal(DRight,eps))
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

