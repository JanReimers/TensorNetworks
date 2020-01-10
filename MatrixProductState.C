#include "MatrixProductState.H"
#include "MatrixProductOperator.H"
#include "oml/cnumeric.h"
#include "oml/vector_io.h"
#include "TensorNetworks/PrimeEigenSolver.H"
#include <iostream>
#include <iomanip>
#include <complex>

using std::cout;
using std::endl;

MatrixProductState::MatrixProductState(int L, int S2, int D)
    : itsL(L)
    , itsS2(S2)
    , itsD(D)
    , itsp(itsS2+1)
{
    itsSites.push_back(new MatrixProductSite(MatrixProductSite::Left,itsp,1,itsD));
    for (int i=1;i<itsL-1;i++)
        itsSites.push_back(new MatrixProductSite(MatrixProductSite::Bulk,itsp,itsD,itsD));
    itsSites.push_back(new MatrixProductSite(MatrixProductSite::Right,itsp,itsD,1));
}

MatrixProductState::~MatrixProductState()
{
    //dtor
}

void MatrixProductState::InitializeWith(MatrixProductSite::State state)
{
    int sgn=1;
    for (SIter i=itsSites.begin();i!=itsSites.end();i++)
    {
        i->InitializeWith(state,sgn);
        sgn*=-1;
    }

}


void MatrixProductState::Normalize(MatrixProductSite::Position LR)
{
    VectorT s; // This get passed from one site to the next.
    if (LR==MatrixProductSite::Left)
    {
        MatrixT Vdagger;// This get passed from one site to the next.
        for (int ia=0;ia<itsL-1;ia++)
        {
            itsSites[ia]->SVDLeft_Normalize(s,Vdagger);
            itsSites[ia+1]->Contract(s,Vdagger);
        }
        itsSites[itsL-1]->ReshapeFromLeft(s.GetHigh());
        double norm=std::real(itsSites[itsL-1]->GetLeftNorm()(1,1));
        itsSites[itsL-1]->Rescale(sqrt(norm));
    }
    else if (LR==MatrixProductSite::Right)
    {
        MatrixT U;// This get passed from one site to the next.
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
void MatrixProductState::Normalize(int isite)
{
    if (isite>0)
    {
        VectorT s; // This get passed from one site to the next.
        MatrixT Vdagger;// This get passed from one site to the next.
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
        MatrixT U;// This get passed from one site to the next.
        for (int ia=itsL-1; ia>isite; ia--)
        {
            itsSites[ia]->SVDRightNormalize(U,s);
            itsSites[ia-1]->Contract(U,s);
        }
        itsSites[isite]->ReshapeFromRight(s.GetHigh());
    }
}


double MatrixProductState::GetOverlap() const
{
    cSIter i=itsSites.begin();
    MatrixT E=i->GetE();
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

    //--------------------------------------------------------
    //
    // Calc Eleft
    // Handle left boundary cases
    //
MatrixProductState::MatrixT MatrixProductState::GetMLeft(int isite) const
{
   assert(isite<itsL);
    MatrixT Eleft;
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
   //--------------------------------------------------------
    //
    // Calc Eright
    // Handle right boundary cases
    //

MatrixProductState::MatrixT MatrixProductState::GetMRight(int isite) const
{
    MatrixT Eright;
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

 MatrixProductState::MatrixT MatrixProductState::GetNeff(int isite) const
 {
//    MatrixT Eleft=GetMLeft(isite);
//    MatrixT ERight=GetMRight(isite);

    return itsSites[isite]->GetNeff(GetMLeft(isite),GetMRight(isite));
 }

bool MatrixProductState::CheckNormalized(int isite,double eps) const
{
    MatrixT Neff=GetNeff(isite);
    int N=Neff.GetNumRows();
    MatrixT I(N,N);
    Unit(I);
    double error=Max(abs(Neff-I));
    if (error>1e-12)
        cout << "Warning: Normalization site=" << isite << "  Neff-I error " << error << endl;
    return error<eps;
}

std::string MatrixProductState::GetNormStatus() const
{
    std::string ret;
    for (cSIter i=itsSites.begin(); i!=itsSites.end(); i++)
        ret+=i->GetNormStatus();
    return ret;
}

void MatrixProductState::Report(std::ostream& os) const
{
    os.precision(3);
    os << "Matrix product state for " << itsL << " lattice sites." << endl;
    os << "  Site  D1  D2  Bond Entropy   #updates  Rank  Sparsisty     Emin      Egap" << endl;
    for (int ia=0; ia<itsL; ia++)
    {
        os << std::setw(3) << ia << "  ";
        itsSites[ia]->Report(os);
        os << endl;
    }
}


MatrixProductState::Matrix6T MatrixProductState::GetEO(int isite, const MPOSite* mpos) const
{
    return itsSites[isite]->GetEO(mpos);
}

double MatrixProductState::ContractHeff(int isite,const Matrix6T& Heff) const
{
    return itsSites[isite]->ContractHeff(Heff);
}
double MatrixProductState::ContractHeff(int isite,const MatrixT& Heff) const
{
    return itsSites[isite]->ContractHeff(Heff);
}

void MatrixProductState::SweepRight(const MatrixProductOperator* mpo,bool quiet)
{
    if (!quiet)
    {
        cout.precision(10);
        cout << "SweepRight entry Norm Status         =" << GetNormStatus() << "  E=" << GetExpectationIterate(mpo)/(itsL-1) << endl;
    }
    for (int ia=0; ia<itsL-1; ia++)
    {
        VectorCT Anew=Refine(mpo,ia);

        VectorT s;
        MatrixT Vdagger;
        itsSites[ia]->Update(Anew);
        itsSites[ia]->SVDLeft_Normalize(s,Vdagger);
        itsSites[ia+1]->Contract(s,Vdagger);
        itsSites[ia]->UpdateCache(mpo->GetSite(ia),GetHLeft_Cache(ia-1),GetHRightCache(ia+1));
        if (!quiet) cout << "SweepRight post constract Norm Status=" << GetNormStatus() << "  E=" << GetExpectationIterate(mpo)/(itsL-1) << endl;
    }
}

void MatrixProductState::SweepLeft(const MatrixProductOperator* mpo,bool quiet)
{
    if (!quiet)
    {
        cout.precision(10);
        cout << "SweepLeft  entry Norm Status         =" << GetNormStatus() << "  E=" << GetExpectationIterate(mpo)/(itsL-1) << endl;
    }
    for (int ia=itsL-1; ia>0; ia--)
    {
        VectorCT Anew=Refine(mpo,ia);

        VectorT s;
        MatrixT U;
        itsSites[ia]->Update(Anew);
        itsSites[ia]->SVDRightNormalize(U,s);
        itsSites[ia-1]->Contract(U,s);
        itsSites[ia]->UpdateCache(mpo->GetSite(ia),GetHLeft_Cache(ia-1),GetHRightCache(ia+1));
        if (!quiet)
            cout << "SweepLeft  post constract Norm Status=" << GetNormStatus() << "  E=" << GetExpectationIterate(mpo)/(itsL-1) << endl;

    }
}

MatrixProductState::Vector3T MatrixProductState::GetHLeft_Cache (int isite) const
{
    Vector3T HLeft(1,1,1,1);
    HLeft(1,1,1)=eType(1.0);
    if (isite>=0)  HLeft=itsSites[isite]->GetHLeft_Cache();
    return HLeft;
}
MatrixProductState::Vector3T MatrixProductState::GetHRightCache(int isite) const
{
    Vector3T HRight(1,1,1,1);
    HRight(1,1,1)=eType(1.0);
    if (isite<itsL)  HRight=itsSites[isite]->GetHRightCache();
    return HRight;
}


MatrixProductState::Matrix6T MatrixProductState::GetHeffIterate   (const MatrixProductOperator* mpo,int isite) const
{
    Vector3T Lcache=GetHLeft_Cache(isite-1);
    Vector3T Rcache=GetHRightCache(isite+1);
#ifdef DEBUG
    Vector3T L=GetEOLeft_Iterate(mpo,isite,false);
    Vector3T R=GetEORightIterate(mpo,isite,false);
    double errorL=Max(abs(L.Flatten()-Lcache.Flatten()));
    double errorR=Max(abs(R.Flatten()-Rcache.Flatten()));
    double eps=1e-12;
    if (errorL>eps || errorR>eps)
    {
        cout << "Warning Heff errors Left,Rigt=" << std::scientific << errorL << " " << errorR << endl;
        cout << "L=" << L  << endl;
        cout << "Lcache(ia-1)=" << Lcache  << endl;
        cout << "Lcache(ia  )=" << GetHLeft_Cache(isite)  << endl;
        cout << "R=" << R  << endl;
        cout << "Rcache(ia+1)=" << Rcache  << endl;
    }
    assert(errorL<eps);
    assert(errorR<eps);

#endif


    const MPOSite* mops=mpo->GetSite(isite);
    assert(mops);
    ipairT Ds=GetDs(isite);
    int D1=Ds.first;
    int D2=Ds.second;

    Matrix6<eType> Heff(itsp,D1,D2,itsp,D1,D2);

    for (int m=0; m<itsp; m++)
        for (int i1=1; i1<=D1; i1++)
            for (int j1=1; j1<=D1; j1++)
            {
                for (int n=0; n<itsp; n++)
                {
                    MatrixT W=mops->GetW(m,n);
                    for (int i2=1; i2<=D2; i2++)
                        for (int j2=1; j2<=D2; j2++)
                        {
                            eType temp(0.0);
                            for (int w1=1; w1<=W.GetNumRows(); w1++)
                                for (int w2=1; w2<=W.GetNumCols(); w2++)
                                {
                                    temp+=Lcache(w1,i1,j1)*W(w1,w2)*Rcache(w2,i2,j2);
                                }

                            Heff(m,i1,i2,n,j1,j2)=temp;
                        }
                }
            }
    return Heff;

}

MatrixProductState::Vector3T MatrixProductState::GetEOLeft_Iterate(const MatrixProductOperator* mpo,int isite, bool cache) const
{
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=0; ia<isite; ia++)
        F=itsSites[ia]->IterateLeft_F(mpo->GetSite(ia),F,cache);
    return F;
}
MatrixProductState::Vector3T MatrixProductState::GetEORightIterate(const MatrixProductOperator* mpo,int isite, bool cache) const
{
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=itsL-1; ia>isite; ia--)
    {
//        cout << "ia=" << ia << "   ";
        F=itsSites[ia]->IterateRightF(mpo->GetSite(ia),F,cache);
    }
    return F;
}

void MatrixProductState::LoadHeffCaches(const MatrixProductOperator* mpo)
{
    GetEOLeft_Iterate(mpo,0,true);
    GetEORightIterate(mpo,0,true);
}

double   MatrixProductState::GetExpectationIterate   (const MatrixProductOperator* mpo) const
{
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=0; ia<itsL; ia++)
        F=itsSites[ia]->IterateLeft_F(mpo->GetSite(ia),F);

    double iE=std::imag(F(1,1,1));
    if (fabs(iE)>1e-10)
        cout << "Warning: MatrixProductState::GetExpectation Imag(E)=" << std::imag(F(1,1,1)) << endl;

    return std::real(F(1,1,1));
}




MatrixProductState::Matrix6T MatrixProductState::GetHeff(const MatrixProductOperator *mpo,int isite) const
{
    Matrix6T NLeft =GetEOLeft (mpo,isite);
    Matrix6T NRight=GetEORight(mpo,isite);
//    cout << "NLeft " << NLeft  << endl;
//    cout << "NRight" << NRight << endl;
//    assert(NLeft .GetNumRows()==1);
 //   assert(NRight.GetNumCols()==1);

    const MPOSite* mops=mpo->GetSite(isite);
    assert(mops);
    ipairT Ds=GetDs(isite);
    int D1=Ds.first;
    int D2=Ds.second;
    int p=itsp;

    Matrix6<eType> Heff(p,D1,D2,p,D1,D2);

    for (int m=0; m<p; m++)
        for (int i1=1; i1<=D1; i1++)
            for (int j1=1; j1<=D1; j1++)
            {
                for (int n=0; n<p; n++)
                {
                    MatrixT W=mops->GetW(m,n);
                    for (int i2=1; i2<=D2; i2++)
                        for (int j2=1; j2<=D2; j2++)
                        {
                            eType temp(0.0);
                            for (int w1=1; w1<=W.GetNumRows(); w1++)
                                for (int w2=1; w2<=W.GetNumCols(); w2++)
                                {
                                    temp+=NLeft(1,1,1,w1,i1,j1)*W(w1,w2)*NRight(w2,i2,j2,1,1,1);
                                }

                            Heff(m,i1,i2,n,j1,j2)=temp;
                        }
                }
            }
    return Heff;
}

MatrixProductState::Matrix6T MatrixProductState::GetEOLeft(const MatrixProductOperator *mpo,int isite) const
{
    Matrix6T NLeft(1,1);
    NLeft.Fill(std::complex<double>(1.0));
    for (int ia=0;ia<isite;ia++)
    { //loop over sites
        NLeft*=GetEO(ia,mpo->GetSite(ia));
//        cout << "MPO Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    return NLeft;
}

MatrixProductState::Matrix6T MatrixProductState::GetEORight(const MatrixProductOperator *mpo,int isite) const
{
    Matrix6T NRight(1,1);
    NRight.Fill(std::complex<double>(1.0));
    for (int ia=itsL-1;ia>isite;ia--)
    { //loop over sites
        Matrix6T temp=NRight;
        Matrix6T E=GetEO(ia,mpo->GetSite(ia));

//        cout << "NRight=" <<  NRight << endl;
//        cout << "E=" <<  E << endl;
//        Matrix6T Et=E*temp;
//        cout << "Et=" <<  Et << endl;
        NRight.ClearLimits();
        NRight=E*=temp;
//        cout << "MPO Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
    }
    return NRight;
}

double MatrixProductState::GetExpectation(const MatrixProductOperator *mpo) const
{
    assert(mpo);
    Matrix6T E(1,1);
    E.Fill(std::complex<double>(1.0));

    // E is now a unit row vector
    for (int isite=0;isite<itsL;isite++)
    { //loop over sites
//        Matrix6T temp=mps->GetEO(isite,itsSites[lbr]);
        E*=GetEO(isite,mpo->GetSite(isite));
//        cout << "E[" << isite << "]=" << endl;
//        E.Dump(cout);
//        cout << "MPO Elimits=" << E.GetLimits() << " lbr=" << lbr << endl;
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
        cout << "Warning: MatrixProductOperator::GetExpectation Imag(E)=" << std::imag(E(1,1,1,1,1,1)) << endl;

    return std::real(E(1,1,1,1,1,1));
}

MatrixProductState::VectorCT MatrixProductState::Refine(const MatrixProductOperator *mpo,int isite) const
{
    double itsEps=1e-12;
    assert(CheckNormalized(isite,1e-11));
    //Matrix6T Heff6=GetHeff(mps,isite); Old version
    Matrix6T Heff6=GetHeffIterate(mpo,isite); //New iterative version

    MatrixT Heff=Heff6.Flatten();
    itsSites[isite]->Analyze(Heff); //Record % non zero elements
    assert(Heff.GetNumRows()==Heff.GetNumCols());
    int N=Heff.GetNumRows();
    Vector<double>  eigenValues(N);
//    int ierr=0;
//    ch(Heff, eigenValues ,true,ierr);
//    assert(ierr==0);

    PrimeEigenSolver<eType> solver(Heff,itsEps);
    solver.Solve(2); //Get lowest two eigen values/states
    eigenValues=solver.GetEigenValues();


    itsSites[isite]->SetEnergies(eigenValues(1),eigenValues(2)-eigenValues(1));
    //cout << "eigenValues=" <<  eigenValues << endl;
    //cout << "eigenVector(1)=" <<  Heff.GetColumn(1) << endl;

    return solver.GetEigenVector(1);
}
