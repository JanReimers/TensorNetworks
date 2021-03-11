#include "Operators/SiteOperatorImp.H"
#include "Operators/OperatorClient.H"
#include "TensorNetworksImp/SpinCalculator.H"
#include "oml/diagonalmatrix.h"

namespace TensorNetworks
{

//
//  Build with Dw=1 identity operators
//
SiteOperatorImp::SiteOperatorImp(int d, MPOForm f)
    : itsd(d)
    , itsLBR(PBulk)
    , itsWs(1,1,dtoS(d),f)
    , itsLeft_Bond(nullptr)
    , itsRightBond(nullptr)
{
    Unit(itsWs);
    SetLimits();
}

//
//  Not covered by MPO tests, try the Expectation tests.
//
SiteOperatorImp::SiteOperatorImp(int d, SpinOperator so) //Construct spin operator
    : SiteOperatorImp(d,FUnit)
{
    itsWs(0,0)=OperatorElement<double>::Create(so,d);
    SetLimits();
}

SiteOperatorImp::SiteOperatorImp(int d,Position lbr, const OperatorClient* H,MPOForm f)
    : itsd(d)
    , itsLBR(lbr)
    , itsWs(H->GetW(f))
    , itsLeft_Bond(nullptr)
    , itsRightBond(nullptr)
{
    switch (f)
    {
    case RegularLower:
        Init_lr(lbr,itsWs.GetNumRows()-1,0);
        break;
    case RegularUpper:
        Init_lr(lbr,0,itsWs.GetNumCols()-1);
        break;
    default:
        assert(false);
    }
}

//
// Build from a trotter decomp.
//
SiteOperatorImp::SiteOperatorImp(int d, Direction lr,Position lbr,const MatrixRT& U, const DiagonalMatrixRT& s)
    : itsd(d)
    , itsLBR(lbr)
    , itsWs(d,expH)
    , itsLeft_Bond(nullptr)
    , itsRightBond(nullptr)
{
    int Dw=s.GetNumRows();
    assert(Dw==d*d);
    assert(U.GetNumCols()==Dw);
    assert(U.GetNumRows()==Dw);

    if (lr==DLeft)
    {
        itsWs.SetChi12(-1,Dw-2,false);
        int i1=1; //Linear index for (m,n) = 1+m+p*n
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,i1++)
                for (int w=1; w<=Dw; w++)
                    itsWs(0,w-1)(m,n)=U(i1,w)*sqrt(s(w,w));
    }
    else if (lr==DRight)
    {
        itsWs.SetChi12(Dw-2,-1,false);
        int i2=1; //Linear index for (m,n) = 1+m+p*n
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,i2++)
                for (int w=1; w<=Dw; w++)
                    itsWs(w-1,0)(m,n)=sqrt(s(w,w))*U(w,i2); //U is actually VT
    }
    else
    {
        // Must have been called with one of the spin decomposition types.
        assert(false);
    }
    Init_lr(lbr,0,0);
}
//
// Construct with W operator. Called by iMPOImp::MakeUnitcelliMPO
//
SiteOperatorImp::SiteOperatorImp(const MatrixOR& W)
    : SiteOperatorImp(W.Getd(),W.GetForm())
{
    itsWs=W;
    SetLimits();
}


SiteOperatorImp::~SiteOperatorImp()
{
    //dtor
}

void SiteOperatorImp::Init_lr(Position lbr, int lindex,int rindex)
{
    switch (lbr)
    {
    case PLeft:
        {
            MatrixRT l(0,0,0,itsWs.GetNumRows()-1);
            Fill(l,0.0);
            l(0,lindex)=1.0;
            itsWs=l*itsWs;
        }
        break;
    case PRight:
        {
            MatrixRT r(0,itsWs.GetNumCols()-1,0,0);
            Fill(r,0.0);
            r(rindex,0)=1.0;
            itsWs=itsWs*r;
        }
        break;
    case PBulk:
        break;
    }

    SetLimits();
}

void SiteOperatorImp::SetNeighbours(OperatorBond* leftBond, OperatorBond* rightBond)
{
    assert(leftBond || rightBond); //At least one needs to be non zero
    itsLeft_Bond=leftBond;
    itsRightBond=rightBond;
    if (!itsLeft_Bond) itsLBR=PLeft;
    if (!itsRightBond) itsLBR=PRight;
    if (itsLBR==PBulk ) assert(itsLeft_Bond && itsRightBond);
}

OperatorBond* SiteOperatorImp::GetBond(Direction lr) const
{
    OperatorBond* ret=0;
    switch(lr)
    {
    case DLeft:
        ret=itsRightBond;
        break;
    case DRight:
        ret=itsLeft_Bond;
        break;
    default:
        assert(false);
    }
    assert(ret);
    return ret;
}

void SiteOperatorImp::SetLimits()
{
    const MatLimits& l=itsWs.GetLimits();
    itsOpRange.resize(itsWs.GetLimits());
    for (index_t w1:itsWs.rows())
        for (index_t w2:itsWs.cols())
            if (fabs(itsWs(w1,w2))>0.0) //TODO should be using and eps~1e-15 here.
                itsOpRange.NonZeroAt(w1,w2);
//    cout << "W=" << itsWs << endl;
//    cout << "Ranges=" << itsOpRange << endl;
}

//void  SiteOperatorImp::AccumulateTruncationError(double err)
//{
//    itsTruncationError=sqrt(itsTruncationError*itsTruncationError+err*err);
//}


void SiteOperatorImp::Product(const SiteOperator* O2)
{
    assert(Getd()==O2->Getd());
    const SiteOperatorImp* O2i(dynamic_cast<const SiteOperatorImp*>(O2));
    assert(O2i);
    assert(itsWs.Getd()==O2i->itsWs.Getd());

    itsWs=TensorProduct(itsWs,O2i->itsWs);
    SetLimits();
}

void SiteOperatorImp::Sum(const SiteOperator* O2, double factor)
{
    assert(Getd()==O2->Getd());
    const SiteOperatorImp* O2i(dynamic_cast<const SiteOperatorImp*>(O2));
    assert(O2i);
    assert(itsWs.Getd()==O2i->itsWs.Getd());
    MatrixOR W2=O2i->itsWs;
    W2*=factor;
    if (itsLBR==PLeft)
        itsWs=TensorSumLeft(itsWs,W2); //Concatenate rows
    else if(itsLBR==PRight)
        itsWs=TensorSumLeft(itsWs,W2); //Stack columns
    else
        itsWs=TensorSum(itsWs,W2);
    SetLimits();
}

char TriTypeToChar(TriType ul)
{
    char ret('?');
    switch (ul)
    {
    case Lower:
        ret='L';
        break;
    case Upper:
        ret='U';
        break;
    case Diagonal:
        ret='D';
        break;
    case Full:
        ret='F';
        break;
    case Row:
        ret='R';
        break;
    case Column:
        ret='C';
        break;
    default :
        ret='?';

    }
    return ret;
}
char FormToChar(MPOForm f)
{
    char ret(' ');
    switch (f)
    {
    case RegularLower:
        ret='L';
        break;
    case RegularUpper:
        ret='U';
        break;
    case expH:
        ret='E';
        break;
    case FUnit:
        ret='1';
        break;
    case FUnknown:
        ret='?';
        break;
    default :
        ret='-';

    }
    return ret;
}

char SiteOperatorImp::GetForm() const
{
    return FormToChar(itsWs.GetForm());
}

void SiteOperatorImp::Report(std::ostream& os) const
{
    os
    << std::setw(3) << itsOpRange.Dw1 << " "
    << std::setw(3) << itsOpRange.Dw2 << "   "
    << " " << std::fixed << std::setprecision(1) << std::setw(5) << GetFrobeniusNorm()
    << " " << std::setw(4) << GetNormStatus(1e-13)
    << " " << std::setw(4) << GetMeasuredShape(1e-13)
    << " " << std::setw(4) << GetForm() // Upper lower
//    << " " << itsDw12.w1_first << " " << itsDw12.w2_last
    ;
}

char SiteOperatorImp::GetMeasuredShape(double eps) const
{
    return TriTypeToChar(itsWs.GetMeasuredShape(eps));
}

double SiteOperatorImp::GetFrobeniusNorm() const
{
    return itsWs.GetFrobeniusNorm();
}

char SiteOperatorImp::GetNormStatus(double eps) const
{
    if (itsOpRange.Dw1*itsOpRange.Dw2>4096) return '?';
    char ret='W'; //Not normalized
    MPOForm f=itsWs.GetForm();
    bool isLeft=false,isRight=false;
    if (f==RegularUpper || f==RegularLower)
    {
        isLeft =itsWs.GetV(DLeft ).IsOrthonormal(DLeft ,eps);
        isRight=itsWs.GetV(DRight).IsOrthonormal(DRight,eps);
    }
    else
    {
        isLeft =itsWs.IsOrthonormal(DLeft ,eps);
        isRight=itsWs.IsOrthonormal(DRight,eps);
    }

    if (isLeft && isRight)
        ret='I';
    else if (isLeft)
        ret='L';
    else if (isRight)
        ret='R';

    return ret;
}


} //namespace

