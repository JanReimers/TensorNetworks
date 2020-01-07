#include "MatrixProductState.H"
#include "MatrixProductOperator.H"
#include <iostream>
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

void MatrixProductState::SweepRight(const MatrixProductOperator* mpo)
{
        cout << "SweepRight entry Norm Status         =" << GetNormStatus() << "  E=" << mpo->GetExpectation(this) << endl;
    for (int ia=0; ia<itsL-1; ia++)
    {
        VectorCT Anew=mpo->Refine(this,ia);

        VectorT s;
        MatrixT Vdagger;
        itsSites[ia]->Update(Anew);
//        cout << "SweepRight post update Norm Status   =" << GetNormStatus() << "  E=" << mpo->GetExpectation(this) << endl;
        itsSites[ia]->SVDLeft_Normalize(s,Vdagger);
//        cout << "SweepRight post SVD Norm Status      =" << GetNormStatus() << "  E=" << mpo->GetExpectation(this) << endl;
        itsSites[ia+1]->Contract(s,Vdagger);
        cout << "SweepRight post constract Norm Status=" << GetNormStatus() << "  E=" << mpo->GetExpectation(this) << endl;


//        double E=mpo->GetExpectation(this)/itsL;
//        double o=GetOverlap();
//        cout << "After Refine1 E(" << ia << ")=" << E << "  Overlap=" << o << endl;
//        itsMPS->Normalize(ia+1);
//        cout << "After Refine2 E=" << E << endl;
    }
}

void MatrixProductState::SweepLeft(const MatrixProductOperator* mpo)
{
        cout << "SweepLeft  entry Norm Status         =" << GetNormStatus() << "  E=" << mpo->GetExpectation(this) << endl;
    for (int ia=itsL-1; ia>0; ia--)
    {
        VectorCT Anew=mpo->Refine(this,ia);

        VectorT s;
        MatrixT U;
        itsSites[ia]->Update(Anew);
//        cout << "SweepLeft  post update Norm Status   =" << GetNormStatus() << "  E=" << mpo->GetExpectation(this) << endl;
        itsSites[ia]->SVDRightNormalize(U,s);
//        cout << "SweepLeft  post SVD Norm Status      =" << GetNormStatus() << "  E=" << mpo->GetExpectation(this) << endl;
        itsSites[ia-1]->Contract(U,s);
        cout << "SweepLeft  post constract Norm Status=" << GetNormStatus() << "  E=" << mpo->GetExpectation(this) << endl;


//        double E=mpo->GetExpectation(this)/itsL;
//        double o=GetOverlap();
//        cout << "After Refine1 E(" << ia << ")=" << E << "  Overlap=" << o << endl;
//        itsMPS->Normalize(ia+1);
//        cout << "After Refine2 E=" << E << endl;
    }
}

MatrixProductState::Matrix6T MatrixProductState::GetHeffIterate   (const MatrixProductOperator* mpo,int isite) const
{
    Vector3T L=GetEOLeft_Iterate(mpo,isite);
    Vector3T R=GetEORightIterate(mpo,isite);
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
                                    temp+=L(w1,i1,j1)*W(w1,w2)*R(w2,i2,j2);
                                }

                            Heff(m,i1,i2,n,j1,j2)=temp;
                        }
                }
            }
    return Heff;

}

MatrixProductState::Vector3T MatrixProductState::GetEOLeft_Iterate(const MatrixProductOperator* mpo,int isite) const
{
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=0; ia<isite; ia++)
        F=itsSites[ia]->IterateLeft_F(mpo->GetSite(ia),F);
    return F;
}
MatrixProductState::Vector3T MatrixProductState::GetEORightIterate(const MatrixProductOperator* mpo,int isite) const
{
    Vector3T F(1,1,1,1);
    F(1,1,1)=eType(1.0);
    for (int ia=itsL-1; ia>isite; ia--)
        F=itsSites[ia]->IterateRightF(mpo->GetSite(ia),F);
    return F;
}


