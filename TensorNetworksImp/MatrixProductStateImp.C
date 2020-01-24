#include "TensorNetworksImp/MatrixProductStateImp.H"
#include "TensorNetworks/Hamiltonian.H"
#include "oml/cnumeric.h"
#include "oml/vector_io.h"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

//-------------------------------------------------------------------------------
//
//  Init/construction zone
//
MatrixProductStateImp::MatrixProductStateImp(int L, int S2, int D)
    : itsL(L)
    , itsS2(S2)
    , itsD(D)
    , itsp(itsS2+1)
{
    itsSites.push_back(new MatrixProductSite(TensorNetworks::Left,itsp,1,itsD));
    for (int i=1;i<itsL-1;i++)
        itsSites.push_back(new MatrixProductSite(TensorNetworks::Bulk,itsp,itsD,itsD));
    itsSites.push_back(new MatrixProductSite(TensorNetworks::Right,itsp,itsD,1));
}

MatrixProductStateImp::~MatrixProductStateImp()
{
    //dtor
}

void MatrixProductStateImp::InitializeWith(TensorNetworks::State state)
{
    int sgn=1;
    for (SIter i=itsSites.begin();i!=itsSites.end();i++)
    {
        i->InitializeWith(state,sgn);
        sgn*=-1;
    }

}

//-------------------------------------------------------------------------------
//
//   Normalization routines
//
void MatrixProductStateImp::Normalize(TensorNetworks::Position LR)
{
    VectorT s; // This get passed from one site to the next.
    if (LR==TensorNetworks::Left)
    {
        MatrixCT Vdagger;// This get passed from one site to the next.
        for (int ia=0;ia<itsL-1;ia++)
        {
            itsSites[ia]->SVDLeft_Normalize(s,Vdagger);
            itsSites[ia+1]->Contract(s,Vdagger);
        }
        itsSites[itsL-1]->ReshapeFromLeft(s.GetHigh());
        double norm=std::real(itsSites[itsL-1]->GetLeftNorm()(1,1));
        itsSites[itsL-1]->Rescale(sqrt(norm));
    }
    else if (LR==TensorNetworks::Right)
    {
        MatrixCT U;// This get passed from one site to the next.
        for (int ia=itsL-1;ia>0;ia--)
        {
            itsSites[ia]->SVDRightNormalize(U,s);
            itsSites[ia-1]->Contract(U,s);
        }
        itsSites[0]->ReshapeFromRight(s.GetHigh());
        double norm=std::real(itsSites[0]->GetRightNorm()(1,1));
        itsSites[0]->Rescale(sqrt(norm));
    }

}

//
//  Mixed canonical  A*A*A*A...A*M(isite)*B*B...B*B
//
void MatrixProductStateImp::Normalize(int isite)
{
    if (isite>0)
    {
        VectorT s; // This get passed from one site to the next.
        MatrixCT Vdagger;// This get passed from one site to the next.
        for (int ia=0; ia<isite; ia++)
        {
            itsSites[ia]->SVDLeft_Normalize(s,Vdagger);
            itsSites[ia+1]->Contract(s,Vdagger);
        }
        itsSites[isite]->ReshapeFromLeft(s.GetHigh());
    }

    if (isite<itsL-1)
    {
        VectorT s; // This get passed from one site to the next.
        MatrixCT U;// This get passed from one site to the next.
        for (int ia=itsL-1; ia>isite; ia--)
        {
            itsSites[ia]->SVDRightNormalize(U,s);
            itsSites[ia-1]->Contract(U,s);
        }
        itsSites[isite]->ReshapeFromRight(s.GetHigh());
    }
}

bool MatrixProductStateImp::CheckNormalized(int isite,double eps) const
{
    MatrixCT Neff=GetNeff(isite);
    int N=Neff.GetNumRows();
    MatrixCT I(N,N);
    Unit(I);
    double error=Max(abs(Neff-I));
    if (error>1e-12)
        cout << "Warning: Normalization site=" << isite << "  Neff-I error " << error << endl;
    return error<eps;
}


//--------------------------------------------------------------------------------------
//
// Find ground state
//
int   MatrixProductStateImp::FindGroundState(const Hamiltonian* hamiltonian, int maxIter, double eps)
{
    Normalize(TensorNetworks::Right);
    LoadHeffCaches(hamiltonian);

    int nSweep=0;
    for (int in=0; in<maxIter; in++)
    {
        SweepRight(hamiltonian,true);
        SweepLeft (hamiltonian,true);
        nSweep++;
        cout << "<E^2>-<E>^2=" << GetSigmaE(hamiltonian) << endl;
        if (GetMaxDeltaE()<eps) break;
    }
    return nSweep;
}


void MatrixProductStateImp::SweepRight(const Hamiltonian* h,bool quiet)
{
    if (!quiet)
    {
        cout.precision(10);
        cout << "SweepRight entry Norm Status         =" << GetNormStatus() << "  E=" << GetExpectationIterate(h)/(itsL-1) << endl;
    }
    for (int ia=0; ia<itsL-1; ia++)
    {
        Refine(h,ia);

        VectorT s;
        MatrixCT Vdagger;
        itsSites[ia]->SVDLeft_Normalize(s,Vdagger);
        itsSites[ia+1]->Contract(s,Vdagger);
        itsSites[ia]->UpdateCache(h->GetSiteOperator(ia),GetHLeft_Cache(ia-1),GetHRightCache(ia+1));
        if (!quiet) cout << "SweepRight post constract Norm Status=" << GetNormStatus() << "  E=" << GetExpectationIterate(h)/(itsL-1) << endl;
    }
}

void MatrixProductStateImp::SweepLeft(const Hamiltonian* h,bool quiet)
{
    if (!quiet)
    {
        cout.precision(10);
        cout << "SweepLeft  entry Norm Status         =" << GetNormStatus() << "  E=" << GetExpectationIterate(h)/(itsL-1) << endl;
    }
    for (int ia=itsL-1; ia>0; ia--)
    {
        Refine(h,ia);

        VectorT s;
        MatrixCT U;
        itsSites[ia]->SVDRightNormalize(U,s);
        itsSites[ia-1]->Contract(U,s);
        itsSites[ia]->UpdateCache(h->GetSiteOperator(ia),GetHLeft_Cache(ia-1),GetHRightCache(ia+1));
        if (!quiet)
            cout << "SweepLeft  post constract Norm Status=" << GetNormStatus() << "  E=" << GetExpectationIterate(h)/(itsL-1) << endl;

    }
}
void MatrixProductStateImp::Refine(const Hamiltonian *h,int isite) const
{
    assert(CheckNormalized(isite,1e-11));
    Matrix6T Heff6=GetHeffIterate(h,isite); //New iterative version
    itsSites[isite]->Refine(Heff6.Flatten());
}


MatrixProductStateImp::Vector3T MatrixProductStateImp::GetHLeft_Cache (int isite) const
{
    Vector3T HLeft(1,1,1,1);
    HLeft(1,1,1)=eType(1.0);
    if (isite>=0)  HLeft=itsSites[isite]->GetHLeft_Cache();
    return HLeft;
}
MatrixProductStateImp::Vector3T MatrixProductStateImp::GetHRightCache(int isite) const
{
    Vector3T HRight(1,1,1,1);
    HRight(1,1,1)=eType(1.0);
    if (isite<itsL)  HRight=itsSites[isite]->GetHRightCache();
    return HRight;
}


MatrixProductStateImp::Matrix6T MatrixProductStateImp::GetHeffIterate   (const Hamiltonian* h,int isite) const
{
    Vector3T Lcache=GetHLeft_Cache(isite-1);
    Vector3T Rcache=GetHRightCache(isite+1);
    return itsSites[isite]->GetHeff(h->GetSiteOperator(isite),Lcache,Rcache);
}
void MatrixProductStateImp::LoadHeffCaches(const Hamiltonian* h)
{
    GetEOLeft_Iterate(h,0,true);
    GetEORightIterate(h,0,true);
}

double  MatrixProductStateImp::GetSigmaE(const Hamiltonian* h) const
{
    double E1=GetExpectationIterate(h);
    double E2=GetExpectation(h,h);
    return E2-E1*E1;
}


double  MatrixProductStateImp::GetMaxDeltaE() const
{
    double MaxDeltaE=0.0;
    for (int ia=0; ia<itsL; ia++)
    {
        double de=fabs(itsSites[ia]->GetIterDE());
        if (de>MaxDeltaE) MaxDeltaE=de;
    }
    return MaxDeltaE;
}




//--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//
double MatrixProductStateImp::GetOverlap() const
{
    cSIter i=itsSites.begin();
    MatrixCT E=i->GetE();
    i++;
    for (;i!=itsSites.end();i++)
        E=i->GetELeft(E);
    assert(E.GetNumRows()==1);
    assert(E.GetNumCols()==1);
    double iE=fabs(std::imag(E(1,1)));
    if (iE>1e-12)
        cout << "Warning MatrixProductState::GetOverlap imag(E)=" << iE << endl;
    return std::real(E(1,1));
}

double   MatrixProductStateImp::GetExpectationIterate   (const Operator* o) const
{
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=0; ia<itsL; ia++)
        F=itsSites[ia]->IterateLeft_F(o->GetSiteOperator(ia),F);

    double iE=std::imag(F(1,1,1));
    if (fabs(iE)>1e-10)
        cout << "Warning: MatrixProductState::GetExpectation Imag(E)=" << std::imag(F(1,1,1)) << endl;

    return std::real(F(1,1,1));
}

double   MatrixProductStateImp::GetExpectation(const Operator* o1,const Operator* o2) const
{
    Vector4T F(1,1,1,1,1);
    F(1,1,1,1)=eType(1.0);
    for (int ia=0; ia<itsL; ia++)
        F=itsSites[ia]->IterateLeft_F(o1->GetSiteOperator(ia),o2->GetSiteOperator(ia),F);

    double iE=std::imag(F(1,1,1,1));
    if (fabs(iE)>1e-10)
        cout << "Warning: MatrixProductState::GetExpectation Imag(E)=" << std::imag(F(1,1,1,1)) << endl;

    return std::real(F(1,1,1,1));
}

double MatrixProductStateImp::GetExpectation(const Operator* o) const
{
    assert(o);
    Matrix6T E(1,1);
    E.Fill(std::complex<double>(1.0));

    // E is now a unit row vector
    for (int isite=0;isite<itsL;isite++)
    { //loop over sites
//        Matrix6T temp=mps->GetEO(isite,itsSites[lbr]);
        E*=itsSites[isite]->GetEO(o->GetSiteOperator(isite));
//        cout << "E[" << isite << "]=" << endl;
//        E.Dump(cout);
//        cout << "o Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    // at this point E is 1xDw so we need to dot it with a unit vector
 //   Matrix6T Unit(itsp,1);
   // Unit.Fill(std::complex<double>(1.0));
   // E*=Unit;

//    cout << "E =" << E << endl;
//    assert(E.GetNumRows()==1);
//    assert(E.GetNumCols()==1);
    double iE=std::imag(E(1,1,1,1,1,1));
    if (fabs(iE)>1e-10)
        cout << "Warning: MatrixProduct::GetExpectation Imag(E)=" << std::imag(E(1,1,1,1,1,1)) << endl;

    return std::real(E(1,1,1,1,1,1));
}

//--------------------------------------------------------------------------------------
//
//    Reporting
//
void MatrixProductStateImp::Report(std::ostream& os) const
{
    os.precision(3);
    os << "Matrix product state for " << itsL << " lattice sites." << endl;
    os << "  Site  D1  D2  Bond Entropy   #updates  Rank  Sparsisty     Emin      Egap    dA" << endl;
    for (int ia=0; ia<itsL; ia++)
    {
        os << std::setw(3) << ia << "  ";
        itsSites[ia]->Report(os);
        os << endl;
    }
}

std::string MatrixProductStateImp::GetNormStatus() const
{
    std::string ret;
    for (cSIter i=itsSites.begin(); i!=itsSites.end(); i++)
        ret+=i->GetNormStatus();
    return ret;
}

//--------------------------------------------------------------------------------------
//
//  Allows unit test classes inside.
//
MatrixProductStateImp::MatrixCT MatrixProductStateImp::GetMLeft(int isite) const
{
   assert(isite<itsL);
    MatrixCT Eleft;
    if (isite==0)
    {
        Eleft.SetLimits(1,1);
        Fill(Eleft,std::complex<double>(1.0));
    }
    else
    {
        Eleft=itsSites[0]->GetE();
    }
    //
    //  Zip from left to right up to isite
    //
//    cout << "ELeft(0)=" << Eleft << endl;
    for (int ia=1;ia<isite;ia++)
    {
            Eleft=itsSites[ia]->GetELeft(Eleft);
 //       cout << "ELeft(" << ia << ")=" << Eleft << endl;
            }
    return Eleft;
}


MatrixProductStateImp::MatrixCT MatrixProductStateImp::GetMRight(int isite) const
{
    MatrixCT Eright;
    if (isite==itsL-1)
    {
        Eright.SetLimits(1,1);
        Fill(Eright,std::complex<double>(1.0));
    }
    else
    {
        Eright=itsSites[itsL-1]->GetE();
    }
    // Zip right to left
    for (int ia=itsL-2;ia>=isite+1;ia--)
        Eright=itsSites[ia]->GetERight(Eright);

    return Eright;
}

 MatrixProductStateImp::MatrixCT MatrixProductStateImp::GetNeff(int isite) const
 {
    return itsSites[isite]->GetNeff(GetMLeft(isite),GetMRight(isite));
 }


MatrixProductStateImp::Vector3T MatrixProductStateImp::GetEOLeft_Iterate(const Operator* o,int isite, bool cache) const
{
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=0; ia<isite; ia++)
        F=itsSites[ia]->IterateLeft_F(o->GetSiteOperator(ia),F,cache);
    return F;
}
MatrixProductStateImp::Vector3T MatrixProductStateImp::GetEORightIterate(const Operator* o,int isite, bool cache) const
{
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=itsL-1; ia>isite; ia--)
    {
//        cout << "ia=" << ia << "   ";
        F=itsSites[ia]->IterateRightF(o->GetSiteOperator(ia),F,cache);
    }
    return F;
}






MatrixProductStateImp::Matrix6T MatrixProductStateImp::GetHeff(const Hamiltonian *h,int isite) const
{
    Matrix6T NLeft =GetEOLeft (h,isite);
    Matrix6T NRight=GetEORight(h,isite);
//    cout << "NLeft " << NLeft  << endl;
//    cout << "NRight" << NRight << endl;
//    assert(NLeft .GetNumRows()==1);
 //   assert(NRight.GetNumCols()==1);
    return itsSites[isite]->GetHeff(h->GetSiteOperator(isite),NLeft,NRight);
}

MatrixProductStateImp::Matrix6T MatrixProductStateImp::GetEOLeft(const Operator *o,int isite) const
{
    Matrix6T NLeft(1,1);
    NLeft.Fill(std::complex<double>(1.0));
    for (int ia=0;ia<isite;ia++)
    { //loop over sites
        NLeft*=itsSites[ia]->GetEO(o->GetSiteOperator(ia));
//        cout << "o Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    return NLeft;
}

MatrixProductStateImp::Matrix6T MatrixProductStateImp::GetEORight(const Operator *o,int isite) const
{
    Matrix6T NRight(1,1);
    NRight.Fill(std::complex<double>(1.0));
    for (int ia=itsL-1;ia>isite;ia--)
    { //loop over sites
        Matrix6T temp=NRight;
        Matrix6T E=itsSites[ia]->GetEO(o->GetSiteOperator(ia));

//        cout << "NRight=" <<  NRight << endl;
//        cout << "E=" <<  E << endl;
//        Matrix6T Et=E*temp;
//        cout << "Et=" <<  Et << endl;
        NRight.ClearLimits();
        NRight=E*=temp;
//        cout << "o Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    return NRight;
}


