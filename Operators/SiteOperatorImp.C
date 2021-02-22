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
    , itsTruncationError(0.0)
    , itsWs(1,1,d,Diagonal)
{
    Unit(itsWs);
    SetLimits();
}

//
//  Not covered by MPO tests, try the Expectation tests.
//
SiteOperatorImp::SiteOperatorImp(int d, SpinOperator so) //Construct spin operator
    : SiteOperatorImp(d)
{
    itsWs(0,0)=OperatorElement<double>::Create(so,d);
    SetLimits();
}
//
//  Build from a OpClient Hamiltonian.
//
SiteOperatorImp::SiteOperatorImp(int d,Position lbr, const OperatorClient* H)
    : SiteOperatorImp(d)
{
    itsWs=H->GetMatrixO(Lower);
    Init_lr(lbr,itsWs.GetNumRows()-1,0);
}

SiteOperatorImp::SiteOperatorImp(int d,Position lbr, const OperatorClient* H,TriType ul)
    : SiteOperatorImp(d)
{
    itsWs=H->GetMatrixO(ul);
    itsWs.SetUpperLower(ul);

    switch (ul)
    {
    case Lower:
        Init_lr(lbr,itsWs.GetNumRows()-1,0);
        break;
    case Upper:
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
    : SiteOperatorImp(d)
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
    : SiteOperatorImp(W.Getd())
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

void SiteOperatorImp::SetNeighbours(SiteOperator* left, SiteOperator* right)
{
    assert(left || right); //At least one needs to be non zero
    itsLeft_Neighbour=dynamic_cast<SiteOperatorImp*>(left);
    itsRightNeighbour=dynamic_cast<SiteOperatorImp*>(right);
    assert(!left  || itsLeft_Neighbour); //if left is nonzero then did the cast work?
    assert(!right || itsRightNeighbour);
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
    const MatLimits& l=itsWs.GetLimits();
    itsOpRange.resize(itsWs.GetLimits());
    for (index_t w1:itsWs.rows())
        for (index_t w2:itsWs.cols())
            if (fabs(itsWs(w1,w2))>0.0) //TODO should be using and eps~1e-15 here.
                itsOpRange.NonZeroAt(w1,w2);
}

void  SiteOperatorImp::AccumulateTruncationError(double err)
{
    itsTruncationError=sqrt(itsTruncationError*itsTruncationError+err*err);
}


void SiteOperatorImp::Product(const SiteOperator* O2)
{
    const SiteOperatorImp* O2i(dynamic_cast<const SiteOperatorImp*>(O2));
    assert(O2i);
    itsWs=TensorProduct(itsWs,O2i->itsWs);
    SetLimits();
}

char SiteOperatorImp::GetUL() const
{
    char ret('?');
    switch (itsWs.GetUpperLower())
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
    default :
        ret='?';

    }
    return ret;
}

void SiteOperatorImp::Report(std::ostream& os) const
{
    os
    << std::setw(3) << itsOpRange.Dw1 << " "
    << std::setw(3) << itsOpRange.Dw2 << "   "
    << std::scientific << std::setprecision(1) << itsTruncationError
    << " " << std::fixed << std::setprecision(1) << std::setw(5) << GetFrobeniusNorm()
    << " " << std::setw(4) << GetNormStatus(1e-13)
    << " " << std::setw(4) << GetUpperLower(1e-13)
    << " " << std::setw(4) << GetUL() // Upper lower
//    << " " << itsDw12.w1_first << " " << itsDw12.w2_last
    ;
}

char SiteOperatorImp::GetUpperLower(double eps) const
{
    char ret=' ';
    if (IsUpperTriangular(itsWs,eps))
    {
        if (ret==' ')
            ret='U';
        else if (ret=='L')
            ret='M'; //Mix
    }
    else if (IsLowerTriangular(itsWs,eps))
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
    return itsWs.GetFrobeniusNorm();
}

char SiteOperatorImp::GetNormStatus(double eps) const
{
    if (itsOpRange.Dw1*itsOpRange.Dw2>4096) return '?';
    char ret='W'; //Not normalized
    {
        MatrixOR V=itsWs.GetV(DLeft);
        if (V.GetNumRows()==0)
            ret='l';
        else if (V.IsOrthonormal(DLeft,eps))
            ret='L';
    }
    if (ret!='l')
    {
        MatrixOR V=itsWs.GetV(DRight);
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

