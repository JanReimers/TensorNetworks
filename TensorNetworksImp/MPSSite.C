#include "TensorNetworksImp/MPSSite.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworks/CheckSpin.H"
//#include "NumericalMethods/PrimeEigenSolver.H"
#include "NumericalMethods/LapackEigenSolver.H"
//#include "oml/minmax.h"
#include "oml/cnumeric.h"
//#include "oml/vector_io.h"
#include "oml/random.h"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

namespace TensorNetworks
{

MPSSite::MPSSite(Position lbr, Bond* leftBond, Bond* rightBond,int d, int D1, int D2)
    : itsLeft_Bond(leftBond)
    , itsRightBond(rightBond)
    , itsd(d)
    , itsD1(D1)
    , itsD2(D2)
    , itsHLeft_Cache(1,1,1,1)
    , itsHRightCache(1,1,1,1)
    , itsLeft_Cache (1,1)
    , itsRightCache (1,1)
    , itsEigenSolver(new LapackEigenSolver<dcmplx>())
    , itsNormStatus(NormStatus::M)
    , itsNumUpdates(0)
    , isFrozen(false)
    , itsEmin(0.0)
    , itsGapE(0.0)
    , itsIterDE(1.0)
    , itsPosition(lbr)
{
    if (lbr==PLeft)
    {
        assert(itsRightBond);
    }
    if (lbr==PRight)
    {
        assert(itsLeft_Bond);
    }

    for (int n=0; n<itsd; n++)
    {
        itsMs.push_back(MatrixCT(D1,D2));
        Fill(itsMs.back(),std::complex<double>(0.0));
    }
    itsHLeft_Cache(1,1,1)=1.0;
    itsHRightCache(1,1,1)=1.0;
    itsLeft_Cache (1,1  )=1.0;
    itsRightCache (1,1  )=1.0;
}

MPSSite::~MPSSite()
{
    delete itsEigenSolver;
}

void MPSSite::CloneState(const MPSSite* psi2)
{
    assert(psi2->itsd==itsd);
    for (int n=0;n<itsd;n++)
        itsMs[n]=psi2->itsMs[n];

    itsD1         =psi2->itsD1;
    itsD2         =psi2->itsD2;
    itsHLeft_Cache=psi2->itsHLeft_Cache;
    itsHRightCache=psi2->itsHRightCache;
    itsLeft_Cache =psi2->itsLeft_Cache;
    itsRightCache =psi2->itsRightCache;
    isFrozen      =psi2->isFrozen;
    itsNormStatus =psi2->itsNormStatus;

}

//
//  This is a tricker than one might expect.  In particular I can't get the OBC vectors
//  to be both left and right normalized
//
void MPSSite::InitializeWith(State state,int sgn)
{
    switch (state)
    {
    case Product :
    {
        Position lbr=WhereAreWe();
        switch(lbr)
        {
        case  PLeft :
        {
            int i=1;
            for (dIterT id=itsMs.begin(); id!=itsMs.end(); id++,i++)
                if (i<=itsD2)
                    (*id)(1,i)=std::complex<double>(sgn); //Left normalized
            break;
        }
        case PRight :
        {
            int i=1;
            for (dIterT id=itsMs.begin(); id!=itsMs.end(); id++,i++)
                if (i<=itsD1)
                    (*id)(i,1)=std::complex<double>(sgn);  //Left normalized
            break;
        }
        case PBulk :
        {
            for (dIterT id=itsMs.begin(); id!=itsMs.end(); id++)
                for (int i=1; i<=Min(itsD1,itsD2); i++)
                    (*id)(i,i)=std::complex<double>(sgn/sqrt(itsd));
            break;
        }
        }
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

void MPSSite::Freeze(double Sz)
{
    isFrozen=true;
    assert(isValidSpin(Sz));
    double S=(static_cast<double>(itsd)-1.0)/2.0;
    int n=Sz+S;
    assert(n>=0);
    assert(n<itsd);
    for (dIterT id=itsMs.begin(); id!=itsMs.end(); id++)
        Fill(*id,dcmplx(0.0));
    itsMs[n](1,1)=1.0;
    itsIterDE=0.0;
}

void MPSSite::NewBondDimensions(int D1, int D2, bool saveData)
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

bool MPSSite::IsNormalized(Direction lr,double eps) const
{
    return IsUnit(GetNorm(lr),eps);
}

bool MPSSite::IsCanonical(Direction lr,double eps) const
{
    return IsUnit(GetCanonicalNorm(lr),eps);
}

char MPSSite::GetNormStatus(double eps) const
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

void MPSSite::Report(std::ostream& os) const
{
    os << std::setw(4)          << itsD1
       << std::setw(4)          << itsD2
       << std::setw(5)          << GetNormStatus(1e-12)
       << std::setw(8)          << itsNumUpdates
       << std::fixed      << std::setprecision(8) << std::setw(13) << itsEmin
       << std::fixed      << std::setprecision(5) << std::setw(10) << itsGapE
       << std::scientific << std::setprecision(1) << std::setw(10) << itsIterDE
       ;
}


//bool MPSSite::IsUnit(const MatrixCT& m,double eps)
//{
//    assert(m.GetNumRows()==m.GetNumCols());
//    int N=m.GetNumRows();
//    MatrixCT I(N,N);
//    Unit(I);
//    return Max(fabs(m-I))<eps;
//}

} //namespace
