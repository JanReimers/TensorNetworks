#include "MatrixProductState.H"
#include <iostream>
#include <complex>

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
    for (SIter i=itsSites.begin();i!=itsSites.end();i++)
        i->InitializeWithProductState();
}

double MatrixProductState::GetOverlap() const
{
    cSIter i=itsSites.begin();
    MatrixT E1=i->GetOverlapTransferMatrix();
//    std::cout << "E1=" << E1 << std::endl;
    i++;
    for (;i!=itsSites.end();i++)
    {
        E1*=i->GetOverlapTransferMatrix();
    }
    assert(E1.GetNumRows()==1);
    assert(E1.GetNumCols()==1);
    assert(std::imag(E1(1,1))==0.0);
    return std::real(E1(1,1));
}

 MatrixProductState::MatrixT MatrixProductState::GetOverlap(int isite) const
 {
    assert(isite<itsL);
    MatrixT Eleft,Eright;
    if (isite==0)
    {
        Eleft.SetLimits(1,1);
        Eleft(1,1)=1.0;
    }
    if (isite==itsL-1)
    {
        Eright.SetLimits(1,1);
        Eright(1,1)=1.0;
    }
    for (int ia=0;ia<isite;ia++)
    {
        if (ia==0)
            Eleft=itsSites[ia]->GetOverlapTransferMatrix();
        else
            Eleft*=itsSites[ia]->GetOverlapTransferMatrix();
    }
    for (int ia=isite+1;ia<itsL;ia++)
    {
        if (ia==isite+1)
            Eright=itsSites[ia]->GetOverlapTransferMatrix();
        else
            Eright*=itsSites[ia]->GetOverlapTransferMatrix();
    }
    const MatrixProductSite* site=itsSites[isite];
    MatrixT Sab=site->GetOverlapMatrix(Eleft,Eright);
    return Sab;
 }
