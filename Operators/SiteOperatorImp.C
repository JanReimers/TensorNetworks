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
    , itsWs(1,1,d,Lower)
{
    Unit(itsWs);
    SetLimits();
}

//
//  Not covered by MPO tests, try the Expectation tests.
//
SiteOperatorImp::SiteOperatorImp(int d, double S, SpinOperator so) //Construct spin operator
    : SiteOperatorImp(d)
{
    itsWs(0,0)=OperatorElement<double>::Create(so,S);
    SetLimits();
}
//
//  Build from a OpClient Hamiltonian.
//
SiteOperatorImp::SiteOperatorImp(int d, const OperatorClient* H)
    : SiteOperatorImp(d)
{
    itsWs=H->GetMatrixO(Lower);
    itsDw=Dw12(itsWs.GetNumRows(),itsWs.GetNumCols());
    SetLimits();
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
        itsWs.SetChi12(-1,Dw-2,false);
        itsDw=Dw12(1,Dw);
        int i1=1; //Linear index for (m,n) = 1+m+p*n
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,i1++)
                for (int w=1; w<=Dw; w++)
                    itsWs(0,w-1)(m,n)=U(i1,w)*sqrt(s(w,w));
    }
    else if (lr==DRight)
    {
        itsWs.SetChi12(Dw-2,-1,false);
        itsDw=Dw12(Dw,1);
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
    SetLimits();
}

//
// Construct with W operator. Called by iMPOImp::MakeUnitcelliMPO
//
SiteOperatorImp::SiteOperatorImp(const MatrixOR& W)
    : SiteOperatorImp(W.Getd())
{
    itsWs=W;
    itsDw=Dw12(itsWs.GetNumRows(),itsWs.GetNumCols());
    SetLimits();
}


SiteOperatorImp::~SiteOperatorImp()
{
    //dtor
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
    itsDw.Dw1=itsWs.GetNumRows();
    itsDw.Dw2=itsWs.GetNumCols();
    itsDw.w1_first.SetLimits(itsDw.Dw2);
    itsDw.w2_last .SetLimits(itsDw.Dw1);
//    Fill(Dw.w1_first,1);
//    Fill(DW.w2_last ,Dw.Dw2);

    Fill(itsDw.w1_first,itsDw.Dw1);
    Fill(itsDw.w2_last ,1);
    for (index_t w1:itsWs.rows())
        for (index_t w2:itsWs.cols())
            if (fabs(itsWs(w1,w2))>0.0) //TOT should be using and eps~1e-15 here.
            {
                if (itsDw.w1_first(w2+1)>w1+1) itsDw.w1_first(w2+1)=w1+1;
                if (itsDw.w2_last (w1+1)<w2+1) itsDw.w2_last (w1+1)=w2+1;
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

//    cout << "itsWOvM=" << itsWOvM.GetLimits() << " " << itsWOvM.GetUpperLower() << endl;
//    cout << "O2i->itsWOvM=" << O2i->itsWOvM.GetLimits()<< " " <<  O2i->itsWOvM.GetUpperLower()  << endl;
    itsWs=TensorProduct(itsWs,O2i->itsWs);
    itsDw=Dw;
    SetLimits();
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
    if (itsDw.Dw1*itsDw.Dw2>4096) return '?';
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

