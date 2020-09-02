#include "TensorNetworksImp/MPSSite.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "oml/minmax.h"
#include "oml/cnumeric.h"
//#include "oml/vector_io.h"
#include "oml/random.h"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

MPSSite::MPSSite(TensorNetworks::Position lbr, Bond* leftBond, Bond* rightBond,int d, int D1, int D2)
    : itsLeft_Bond(leftBond)
    , itsRightBond(rightBond)
    , itsd(d)
    , itsD1(D1)
    , itsD2(D2)
    , itsHLeft_Cache(1,1,1,1)
    , itsHRightCache(1,1,1,1)
    , itsLeft_Cache (1,1)
    , itsRightCache (1,1)
    , itsEigenSolver()
    , itsNumUpdates(0)
    , isFrozen(false)
    , itsEmin(0.0)
    , itsGapE(0.0)
    , itsIterDE(1.0)
    , itsPosition(lbr)
{
    if (lbr==TensorNetworks::PLeft)
    {
        assert(itsRightBond);
    }
    if (lbr==TensorNetworks::PRight)
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
    //dtor
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

}

//
//  This is a tricker than one might expect.  In particular I can't get the OBC vectors
//  to be both left and right normalized
//
void MPSSite::InitializeWith(TensorNetworks::State state,int sgn)
{
    switch (state)
    {
    case TensorNetworks::Product :
    {
        TensorNetworks::Position lbr=WhereAreWe();
        switch(lbr)
        {
        case  TensorNetworks::PLeft :
        {
            int i=1;
            for (dIterT id=itsMs.begin(); id!=itsMs.end(); id++,i++)
                if (i<=itsD2)
                    (*id)(1,i)=std::complex<double>(sgn); //Left normalized
            break;
        }
        case TensorNetworks::PRight :
        {
            int i=1;
            for (dIterT id=itsMs.begin(); id!=itsMs.end(); id++,i++)
                if (i<=itsD1)
                    (*id)(i,1)=std::complex<double>(sgn);  //Left normalized
            break;
        }
        case TensorNetworks::PBulk :
        {
            for (dIterT id=itsMs.begin(); id!=itsMs.end(); id++)
                for (int i=1; i<=Min(itsD1,itsD2); i++)
                    (*id)(i,i)=std::complex<double>(sgn/sqrt(itsd));
            break;
        }
        }
        break;
    }
    case TensorNetworks::Random :
    {
        for (dIterT id=itsMs.begin(); id!=itsMs.end(); id++)
        {
            FillRandom(*id);
            (*id)*=1.0/sqrt(itsd*itsD1*itsD2); //Try and keep <psi|psi>~O(1)
        }
        break;
    }
    case TensorNetworks::Neel :
    {
        for (dIterT id=itsMs.begin(); id!=itsMs.end(); id++)
            Fill(*id,eType(0.0));
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
#ifdef DEBUG
    double ipart;
    double frac=std::modf(2.0*Sz,&ipart);
    assert(frac==0.0);
#endif
    double S=(static_cast<double>(itsd)-1.0)/2.0;
    int n=Sz+S;
    assert(n>=0);
    assert(n<itsd);
    for (dIterT id=itsMs.begin(); id!=itsMs.end(); id++)
        Fill(*id,eType(0.0));
    itsMs[n](1,1)=1.0;
    itsIterDE=0.0;
}

void MPSSite::SVDNormalize(TensorNetworks::Direction lr)
{
    // Handle edge cases first
    if (lr==TensorNetworks::DRight && !itsLeft_Bond)
    {
        assert(itsRightBond);
        assert(itsD1==1);
        int newD2=Max(itsRightBond->GetRank(),itsd); //Don't shrink below p
        if (newD2<itsD2) Reshape(itsD1,newD2,true); //But also don't grow D2
        Rescale(sqrt(std::real(GetNorm(lr)(1,1))));
        return;
    }
    if(lr==TensorNetworks::DLeft && !itsRightBond)
    {
        assert(itsLeft_Bond);
        assert(itsD2==1);
        int newD1=Max(itsLeft_Bond->GetRank(),itsd); //Don't shrink below p
        if (newD1<itsD1) Reshape(newD1,itsD2,true); //But also don't grow D1
        Rescale(sqrt(std::real(GetNorm(lr)(1,1))));
        return;
    }

    //We are in the bulk
    VectorT s; // This get passed from one site to the next.
    MatrixCT A=Reshape(lr);
    int N=Min(A.GetNumRows(),A.GetNumCols());
    s.SetLimits(N);
    MatrixCT V(N,A.GetNumCols());
    CSVDecomp(A,s,V); //Solves A=U * s * Vdagger  returns V not Vdagger

    MatrixCT UV;// This get transferred through the bond to a neighbouring site.

    switch (lr)
    {
    case TensorNetworks::DRight:
    {
        UV=A;
        Reshape(lr,Transpose(conj(V)));  //A is now Vdagger
        break;
    }
    case TensorNetworks::DLeft:
    {
        UV=Transpose(conj(V)); //Set Vdagger
        Reshape(lr,A);  //A is now U
        break;
    }
    }
    GetBond(lr)->SVDTransfer(lr,s,UV);
}

void MPSSite::SVDNormalize(TensorNetworks::Direction lr, int Dmax, double epsMin)
{
    // Handle edge cases first
    if (lr==TensorNetworks::DRight && !itsLeft_Bond)
    {
        assert(itsRightBond);
        int newD2=itsRightBond->GetRank();
        Reshape(itsD1,newD2,true);
        Rescale(sqrt(std::real(GetNorm(lr)(1,1))));
        return;
    }
    if(lr==TensorNetworks::DLeft && !itsRightBond)
    {
        assert(itsLeft_Bond);
        int newD1=itsLeft_Bond->GetRank();
        Reshape(newD1,itsD2,true);
        Rescale(sqrt(std::real(GetNorm(lr)(1,1))));
        return;
    }

    //We are in the bulk
    VectorT s; // This get passed from one site to the next.
    MatrixCT A=Reshape(lr);
    int N=Min(A.GetNumRows(),A.GetNumCols());
    s.SetLimits(N);
    MatrixCT V(N,A.GetNumCols());
    CSVDecomp(A,s,V); //Solves A=U * s * Vdagger  returns V not Vdagger
    // At this point we have N singular values but we only Dmax of them or only the ones >=epsMin;
    int D=Dmax>0 ? Min(N,Dmax) : N; //Ignore Dmax if it is 0
    // Shrink so that all s(is<=D)>=epsMin;
    for (int is=D; is>=1; is--)
        if (s(is)>epsMin)
        {
            D=is;
            break;
        }
//    cout << "Smin=" << s(D) << "  Sum of rejected singular values=" << Sum(s.SubVector(D+1,s.size())) << endl;
//    cout << "Before compression Sum s=" << Sum(s) << endl;
    double Sums=Sum(s);
    assert(Sums>0.0);
    s.SetLimits(D,true);  // Resize s
    A.SetLimits(A.GetNumRows(),D,true);
    V.SetLimits(V.GetNumRows(),D,true);
    assert(Sum(s)>0.0);
    double rescaleS=Sums/Sum(s);
    s*=rescaleS;
//    cout << "After compression  Sum s=" << Sum(s) << endl;

    MatrixCT UV;// This get transferred through the bond to a neighbouring site.
    switch (lr)
    {
        case TensorNetworks::DRight:
        {
            UV=A;
            Reshape(lr,Transpose(conj(V)));  //A is now Vdagger
            break;
        }
        case TensorNetworks::DLeft:
        {
            UV=Transpose(conj(V)); //Set Vdagger
            Reshape(lr,A);  //A is now U
            break;
        }
    }
    assert(GetNormStatus(1e-12)!='M');
    GetBond(lr)->SVDTransfer(lr,s,UV);
}

void MPSSite::Rescale(double norm)
{
    for (int n=0; n<itsd; n++) itsMs[n]/=norm;
}

char MPSSite::GetNormStatus(double eps) const
{
//    StreamableObject::SetToPretty();
//    for (int id=0;id<itsd; id++)
//        cout << "A[" << id << "]=" << itsAs[id] << endl;
    char ret;
    if (IsNormalized(TensorNetworks::DLeft,eps))
    {
        if (IsNormalized(TensorNetworks::DRight,eps))
            ret='I'; //This should be rare
        else
            ret='A';
    }
    else if (IsNormalized(TensorNetworks::DRight,eps))
        ret='B';
    else
        ret='M';

    return ret;
}

bool MPSSite::SetCanonicalBondDimensions(int maxAllowedD1,int maxAllowedD2)
{
    bool reshape=false;
    if (itsD1>maxAllowedD1 || itsD2 >maxAllowedD2)
    {
        assert(itsD1>=maxAllowedD1);
        assert(itsD2>=maxAllowedD2);
        Reshape(maxAllowedD1,maxAllowedD2,true);
        reshape=true;
    }
    return reshape;
}

void MPSSite::Report(std::ostream& os) const
{
    os << std::setprecision(3)
       << std::setw(4) << itsD1
       << std::setw(4)  << itsD2 << std::fixed
       << std::setw(5)  << itsNumUpdates << "      "
       << std::setprecision(7)
       << std::setw(9)  << itsEmin << "     " << std::setprecision(4)
       << std::setw(5)  << itsGapE << "   " << std::scientific
       << std::setw(5)  << itsIterDE << "  "
       ;
}

bool MPSSite::IsNormalized(TensorNetworks::Direction lr,double eps) const
{
    return IsUnit(GetNorm(lr),eps);
}

bool MPSSite::IsUnit(const MatrixCT& m,double eps)
{
    assert(m.GetNumRows()==m.GetNumCols());
    int N=m.GetNumRows();
    MatrixCT I(N,N);
    Unit(I);
    return Max(abs(m-I))<eps;
}



void MPSSite::Refine(const MatrixCT& Heff,const Epsilons& eps)
{
    assert(!isFrozen);
    assert(Heff.GetNumRows()==Heff.GetNumCols());
    int N=Heff.GetNumRows();
    Vector<double>  eigenValues(N);
    itsEigenSolver.Solve(Heff,2,eps); //Get lowest two eigen values/states

    eigenValues=itsEigenSolver.GetEigenValues();

    itsIterDE=eigenValues(1)-itsEmin;
    itsEmin=eigenValues(1);
    itsGapE=eigenValues(2)-eigenValues(1);
    Update(itsEigenSolver.GetEigenVector(1));
}

void MPSSite::Update(const VectorCT& newAs)
{
    Vector3<eType> As(itsd,itsD1,itsD2,newAs); //Unflatten
    for (int m=0; m<itsd; m++)
        for (int i1=1; i1<=itsD1; i1++)
            for (int i2=1; i2<=itsD2; i2++)
                itsMs[m](i1,i2)=As(m,i1,i2);

    itsNumUpdates++;
}

void MPSSite::UpdateCache(const SiteOperator* so, const Vector3T& HLeft, const Vector3T& HRight)
{
    itsHLeft_Cache=IterateLeft_F(so,HLeft);
    itsHRightCache=IterateRightF(so,HRight);
}

void MPSSite::UpdateCache(const MPSSite* Psi2,const MatrixCT& Left, const MatrixCT& Right)
{
    itsLeft_Cache =IterateLeft_F(Psi2,Left);
    itsRightCache =IterateRightF(Psi2,Right);
}

// Get this site as close to psi as possible.  In the docs this site is psi^tilda
void MPSSite::Optimize(const MPSSite* psi, const MatrixCT& L, const MatrixCT& R)
{
    assert(itsd==psi->itsd);
    cout.precision(10);
    for (int n=0; n<itsd;n++)
    {
        MatrixCT Anew=ContractLRM(psi->itsMs[n],L,R);
//        cout << "A-Anew " << std::fixed << Max(abs(itsAs[n]-Anew)) << endl;
        itsMs[n]=Anew;
        }
}

MPSSite::MatrixCT MPSSite::ContractLRM(const MatrixCT& M, const MatrixCT& L, const MatrixCT& R) const
{
//    cout << "D1,D2=" << itsD1 << " " << itsD2 << endl;
//    cout << "ContractLRM L=" << L.GetLimits() << endl;
//    cout << "ContractLRM R=" << R.GetLimits() << endl;
    assert(R.GetNumRows()==itsD2);
    assert(L.GetNumRows()==itsD1);

    MatrixCT RM=Contract_RM(R,M);
    assert(RM.GetNumCols()==L.GetNumCols());
    assert(RM.GetNumRows()==itsD2);

    MatrixCT M_tilde(itsD1,itsD2);
    Fill(M_tilde,eType(0.0));

//    cout << "ContractLRM RM=" << RM.GetLimits() << endl;
    for (int i1=1; i1<=itsD1; i1++)
        for (int i2=1; i2<=itsD2; i2++)
            for (int j1=1; j1<=L.GetNumCols(); j1++)
                M_tilde(i1,i2)+=L(i1,j1)*RM(i2,j1);

    return M_tilde;
}

MPSSite::MatrixCT MPSSite::Contract_RM(const MatrixCT& R, const MatrixCT& M) const
{
    assert(R.GetNumCols()==M.GetNumCols());
    assert(R.GetNumRows()==itsD2);
    MatrixCT RM(R.GetNumRows(),M.GetNumRows());
    Fill(RM,eType(0.0));
    for (int i2=1; i2<=R.GetNumRows(); i2++)
        for (int j1=1; j1<=M.GetNumRows(); j1++)
            for (int j2=1; j2<=R.GetNumCols(); j2++)
                RM(i2,j1)+=R(i2,j2)*M(j1,j2);
    return RM;
}
