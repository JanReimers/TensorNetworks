#include "TensorNetworksImp/MatrixProductSite.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "oml/minmax.h"
#include "oml/cnumeric.h"
#include "oml/vector_io.h"
#include "oml/random.h"
#include <iostream>

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
    , itsHeffDensity(0)
    , itsEmin(0.0)
    , itsGapE(0.0)
    , itsIterDE(1.0)
    , itsPosition(lbr)
{
    if (lbr==TensorNetworks::Left)
    {
        assert(itsRightBond);
    }
    if (lbr==TensorNetworks::Right)
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
            case  TensorNetworks::Left :
                {
                    int i=1;
                    for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++,i++)
                        if (i<=itsD2)
                            (*ip)(1,i)=std::complex<double>(sgn); //Left normalized
                    break;
                }
            case TensorNetworks::Right :
                {
                    int i=1;
                    for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++,i++)
                        if (i<=itsD1)
                            (*ip)(i,1)=std::complex<double>(sgn);  //Left normalized
                    break;
                }
            case TensorNetworks::Bulk :
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
                FillRandom(*ip);
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

void MatrixProductSite::SVDLeft_Normalize(VectorT& s, MatrixCT& Vdagger)
{

    MatrixCT A=ReshapeLeft();
    //
    //  Set up and do SVD
    //
    int N=Min(A.GetNumRows(),A.GetNumCols());
    s.SetLimits(N);
    MatrixCT V(N,A.GetNumCols());
    CSVDecomp(A,s,V); //Solves A=U * s * Vdagger  returns V not Vdagger
    Vdagger.SetLimits(0,0);  //Wipe out old data;
    Vdagger=Transpose(conj(V));
    //
    //  Extract As from U
    //
    ReshapeLeft(A);  //A is now U
    if (itsRightBond) itsRightBond->SetSingularValues(s);
}

void MatrixProductSite::SVDRightNormalize(MatrixCT& U, VectorT& s)
{
    MatrixCT A=ReshapeRight();
    int N=Min(A.GetNumRows(),A.GetNumCols());
    s.SetLimits(N);
    MatrixCT V(N,A.GetNumCols());
    CSVDecomp(A,s,V); //Solves A=U * s * Vdagger  returns V not Vdagger
    U.SetLimits(0,0);  //Wipe out old data;
    U=A;
    //
    //  Extract Bs from U
    //
    ReshapeRight(Transpose(conj(V)));  //A is now Vdagger
    assert(itsLeft_Bond);
    if (itsLeft_Bond) itsLeft_Bond->SetSingularValues(s);
}

void MatrixProductSite::ReshapeFromLeft (int D1)
{
    Reshape(   D1,itsD2,true);
}

void MatrixProductSite::ReshapeFromRight(int D2)
{
    Reshape(itsD1,   D2,true);
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
    if (IsLeftNormalized(eps))
    {
        if (IsRightNormalized(eps))
            ret="I"; //This should be rare
        else
            ret="A";
    }
    else
        if (IsRightNormalized(eps))
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
    << std::setw(5)  << itsHeffDensity << "   " << std::setprecision(7)
    << std::setw(9)  << itsEmin << "     " << std::setprecision(4)
    << std::setw(5)  << itsGapE << "   " << std::scientific
    << std::setw(5)  << itsIterDE << "  "
    ;
}


bool MatrixProductSite::IsLeftNormalized(double eps) const
{
    return IsUnit(GetLeftNorm(),eps);
}
bool MatrixProductSite::IsRightNormalized(double eps) const
{
    return IsUnit(GetRightNorm(),eps);
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

