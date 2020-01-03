#include "MatrixProductState.H"
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
        for (SIter i=itsSites.begin(); i!=itsSites.end(); i++)
            i->SVDLeft_Normalize(s,Vdagger);
    }
    else if (LR==MatrixProductSite::Right)
    {
        MatrixT U;// This get passed from one site to the next.
        for (rSIter i=itsSites.rbegin(); i!=itsSites.rend(); i++)
            i->SVDRightNormalize(U,s);
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
            itsSites[ia]->SVDLeft_Normalize(s,Vdagger);
        itsSites[isite]->ReshapeFromLeft(s.GetHigh());
    }

    if (isite<itsL-1)
    {
        VectorT s; // This get passed from one site to the next.
        MatrixT U;// This get passed from one site to the next.
        for (int ia=itsL-1; ia>isite; ia--)
            itsSites[ia]->SVDRightNormalize(U,s);
        itsSites[isite]->ReshapeFromRight(s.GetHigh());
    }
}


double MatrixProductState::GetOverlap() const
{
    cSIter i=itsSites.begin();
    MatrixT E=i->GetOverlapTransferMatrix();
    i++;
    for (;i!=itsSites.end();i++)
        E=i->GetOverlapTransferMatrixLeft(E);
    assert(E.GetNumRows()==1);
    assert(E.GetNumCols()==1);
    assert(fabs(std::imag(E(1,1)))<1.0e-14);
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
        Eleft=itsSites[0]->GetOverlapTransferMatrix();
    }
    //
    //  Zip from left to right up to isite
    //
//    cout << "ELeft(0)=" << Eleft << endl;
    for (int ia=1;ia<isite;ia++)
    {
            Eleft=itsSites[ia]->GetOverlapTransferMatrixLeft(Eleft);
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
        Eright=itsSites[itsL-1]->GetOverlapTransferMatrix();
    }
    // Zip right to left
    for (int ia=itsL-2;ia>=isite+1;ia--)
        Eright=itsSites[ia]->GetOverlapTransferMatrixRight(Eright);

    return Eright;
}

 MatrixProductState::MatrixT MatrixProductState::GetOverlap(int isite) const
 {
//    MatrixT Eleft=GetMLeft(isite);
//    MatrixT ERight=GetMRight(isite);

    return itsSites[isite]->GetOverlapMatrix(GetMLeft(isite),GetMRight(isite));
 }


MatrixProductState::Matrix6T MatrixProductState::GetE(int isite, const MPOSite* mpos) const
{
    return itsSites[isite]->GetE(mpos);
}

double MatrixProductState::ConstractHeff(int isite,const Matrix6T& Heff) const
{
    return itsSites[isite]->ConstractHeff(Heff);
}


