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
    itsSites.push_back(new MatrixProductSite(itsp,1,itsD));
    for (int i=1;i<itsL-1;i++)
        itsSites.push_back(new MatrixProductSite(itsp,itsD,itsD));
    itsSites.push_back(new MatrixProductSite(itsp,itsD,1));
}

MatrixProductState::~MatrixProductState()
{
    //dtor
}

void MatrixProductState::InitializeWithProductState()
{
    int sgn=1;
    for (SIter i=itsSites.begin();i!=itsSites.end();i++)
    {
        i->InitializeWithProductState(sgn);
        sgn*=-1;
    }
}
void MatrixProductState::InitializeWithRandomState()
{
    for (SIter i=itsSites.begin();i!=itsSites.end();i++)
        i->InitializeWithRandomState();
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



MatrixProductState::Matrix6T MatrixProductState::GetEO(int isite, const MPOSite* mpos) const
{
    return itsSites[isite]->GetEO(mpos);
}

double MatrixProductState::ContractHeff(int isite,const Matrix6T& Heff) const
{
    return itsSites[isite]->ContractHeff(Heff);
}

void MatrixProductState::SweepRight(const MatrixProductOperator* mpo)
{
    for (int ia=0; ia<itsL-1; ia++)
    {
        VectorCT Anew=mpo->Refine(this,ia);

        VectorT s;
        MatrixT Vdagger;
        itsSites[ia]->Update(Anew);
        itsSites[ia]->SVDLeft_Normalize(s,Vdagger);
        itsSites[ia+1]->Contract(s,Vdagger);


        double E=mpo->GetExpectation(this)/itsL;
        double o=GetOverlap();
        cout << "After Refine1 E(" << ia << ")=" << E << "  Overlap=" << o << endl;
//        itsMPS->Normalize(ia+1);
//        cout << "After Refine2 E=" << E << endl;
    }
}

void MatrixProductState::SweepLeft(const MatrixProductOperator* mpo)
{
    for (int ia=itsL-1; ia>0; ia--)
    {
        VectorCT Anew=mpo->Refine(this,ia);

        VectorT s;
        MatrixT U;
        itsSites[ia]->Update(Anew);
        itsSites[ia]->SVDRightNormalize(U,s);
        itsSites[ia-1]->Contract(U,s);


        double E=mpo->GetExpectation(this)/itsL;
        double o=GetOverlap();
        cout << "After Refine1 E(" << ia << ")=" << E << "  Overlap=" << o << endl;
//        itsMPS->Normalize(ia+1);
//        cout << "After Refine2 E=" << E << endl;
    }
}

