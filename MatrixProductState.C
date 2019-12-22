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
    typedef MatrixProductSite::MatrixT MatrixT;
    MatrixT E1,E2;
    cSIter i=itsSites.begin();
    E1=i->GetOverlap();
//    std::cout << "E1=" << E1 << std::endl;
    i++;
    for (;i!=itsSites.end();i++)
    {
        E2=i->GetOverlap();
//        std::cout << "  E2=" << E2 << std::endl;
        MatrixT E3=E1*E2;
//        std::cout << "E3=" << E3 << std::endl;
        E1.SetLimits(E3.GetLimits());
        E1=E3;
    }
    assert(E1.GetNumRows()==1);
    assert(E1.GetNumCols()==1);
    assert(std::imag(E1(1,1))==0.0);
    return std::real(E1(1,1));
}
