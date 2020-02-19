#include "TensorNetworksImp/MatrixProductSite.H"
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

MatrixProductSite::MatrixProductSite(TensorNetworks::Position lbr, Bond* leftBond, Bond* rightBond,int p, int D1, int D2)
    : itsLeft_Bond(leftBond)
    , itsRightBond(rightBond)
    , itsp(p)
    , itsD1(D1)
    , itsD2(D2)
    , itsHLeft_Cache(1,1,1,1)
    , itsHRightCache(1,1,1,1)
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

    for (int ip=0;ip<itsp;ip++)
    {
        itsAs.push_back(MatrixCT(D1,D2));
        Fill(itsAs.back(),std::complex<double>(0.0));
    }
    itsHLeft_Cache(1,1,1)=1.0;
    itsHRightCache(1,1,1)=1.0;
}

MatrixProductSite::~MatrixProductSite()
{
    //dtor
}

//
//  This is a tricker than one might expect.  In particular I can't get the OBC vectors
//  to be both left and right normalized
//
void MatrixProductSite::InitializeWith(TensorNetworks::State state,int sgn)
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
                    for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++,i++)
                        if (i<=itsD2)
                            (*ip)(1,i)=std::complex<double>(sgn); //Left normalized
                    break;
                }
            case TensorNetworks::PRight :
                {
                    int i=1;
                    for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++,i++)
                        if (i<=itsD1)
                            (*ip)(i,1)=std::complex<double>(sgn);  //Left normalized
                    break;
                }
            case TensorNetworks::PBulk :
                {
                    for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
                        for (int i=1; i<=Min(itsD1,itsD2); i++)
                            (*ip)(i,i)=std::complex<double>(sgn/sqrt(itsp));
                    break;
                }
            }
            break;
        }
    case TensorNetworks::Random :
        {
            for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
            {
                FillRandom(*ip);
                (*ip)*=1.0/sqrt(itsp*itsD1*itsD2); //Try and keep <psi|psi>~O(1)
            }
            break;
        }
    case TensorNetworks::Neel :
        {
            for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
                Fill(*ip,eType(0.0));
            if (sgn== 1)
                itsAs[0     ](1,1)=1.0;
            if (sgn==-1)
                itsAs[itsp-1](1,1)=1.0;

            break;
        }
    }
}

void MatrixProductSite::Freeze(double Sz)
{
    isFrozen=true;
#ifdef DEBUG
    double ipart;
    double frac=std::modf(2.0*Sz,&ipart);
    assert(frac==0.0);
#endif
    double S=(static_cast<double>(itsp)-1.0)/2.0;
    int n=Sz+S;
    assert(n>=0);
    assert(n<itsp);
    for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
        Fill(*ip,eType(0.0));
    itsAs[n](1,1)=1.0;
    itsIterDE=0.0;
}

void MatrixProductSite::SVDNormalize(TensorNetworks::Direction lr)
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

void MatrixProductSite::Rescale(double norm)
{
    for (int n=0;n<itsp;n++) itsAs[n]/=norm;
}

std::string MatrixProductSite::GetNormStatus(double eps) const
{
//    StreamableObject::SetToPretty();
//    for (int ip=0;ip<itsp; ip++)
//        cout << "A[" << ip << "]=" << itsAs[ip] << endl;
    std::string ret;
    if (IsNormalized(TensorNetworks::DLeft,eps))
    {
        if (IsNormalized(TensorNetworks::DRight,eps))
            ret="I"; //This should be rare
        else
            ret="A";
    }
    else
        if (IsNormalized(TensorNetworks::DRight,eps))
            ret="B";
        else
            ret="M";

    ret+=std::to_string(itsNumUpdates);
    return ret;
}

void MatrixProductSite::Report(std::ostream& os) const
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

bool MatrixProductSite::IsNormalized(TensorNetworks::Direction lr,double eps) const
{
    return IsUnit(GetNorm(lr),eps);
}

bool MatrixProductSite::IsUnit(const MatrixCT& m,double eps)
{
    assert(m.GetNumRows()==m.GetNumCols());
    int N=m.GetNumRows();
    MatrixCT I(N,N);
    Unit(I);
    return Max(abs(m-I))<eps;
}



void MatrixProductSite::Refine(const MatrixCT& Heff,const Epsilons& eps)
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

void MatrixProductSite::Update(const VectorCT& newAs)
{
    Vector3<eType> As(itsp,itsD1,itsD2,newAs); //Unflatten
    for (int m=0; m<itsp; m++)
        for (int i1=1; i1<=itsD1; i1++)
            for (int i2=1; i2<=itsD2; i2++)
                itsAs[m](i1,i2)=As(m,i1,i2);

    itsNumUpdates++;
}

void MatrixProductSite::UpdateCache(const SiteOperator* so, const Vector3T& HLeft, const Vector3T& HRight)
{
    itsHLeft_Cache=IterateLeft_F(so,HLeft);
    itsHRightCache=IterateRightF(so,HRight);
}
