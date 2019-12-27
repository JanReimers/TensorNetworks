#include "MatrixProductSite.H"
#include "oml/minmax.h"
#include <complex>
#include <iostream>

using std::cout;
using std::endl;

MatrixProductSite::MatrixProductSite(int p, int D1, int D2)
    : itsp(p)
    , itsD1(D1)
    , itsD2(D2)
{
    for (int ip=0;ip<itsp;ip++)
    {
        itsAs.push_back(MatrixT(D1,D2));
        Fill(itsAs.back(),std::complex<double>(0.0));
    }
}

MatrixProductSite::~MatrixProductSite()
{
    //dtor
}

//
//  This is a tricker than one might expect.  In particular I can't get the OBC vectors
//  to be both left and right normalized
//
void MatrixProductSite::InitializeWithProductState(int sgn)
{
    if (itsAs[0].GetNumRows()==1)
    {
        int i=1;
        for (pIterT ip=itsAs.begin();ip!=itsAs.end();ip++,i++)
            if (i<=itsD2) (*ip)(1,i)=std::complex<double>(sgn); //Left normalized
    }
    else if (itsAs[0].GetNumCols()==1)
    {
        int i=1;
        for (pIterT ip=itsAs.begin();ip!=itsAs.end();ip++,i++)
            if (i<=itsD1)(*ip)(i,1)=std::complex<double>(sgn);  //Left normalized
    }
    else
    {
        for (pIterT ip=itsAs.begin();ip!=itsAs.end();ip++)
            for (int i=1;i<=Min(itsD1,itsD2);i++)
                (*ip)(i,i)=std::complex<double>(sgn/sqrt(itsp));
    }
}

//
//  Sum_ip A^t(ip) * A(ip)
//
MatrixProductSite::MatrixT MatrixProductSite::GetLeftNorm() const
{
    MatrixT ret(itsD2,itsD2);
    Fill(ret,std::complex<double>(0.0));
    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
        ret+=conj(Transpose((*ip)))*(*ip);
    return ret;
}
//
//  Sum_ip A(ip)*A^t(ip)
//
MatrixProductSite::MatrixT MatrixProductSite::GetRightNorm() const
{
    MatrixT ret(itsD1,itsD1);
    Fill(ret,std::complex<double>(0.0));
    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
        ret+=(*ip)*conj(Transpose((*ip)));
    return ret;
}
//  This is for boundary sites only
//
//  E(1,i,j)=Sum{n,A^t(n;1,i)*A(n;j,1)}=Sum{n,A^*(n;i,1)*A(n;j,1)}
//
MatrixProductSite::MatrixT MatrixProductSite::GetOverlapTransferMatrix() const
{
    assert(GetLimits().GetNumRows()==1 || GetLimits().GetNumCols()==1);

    int D=GetLimits().GetNumRows()==1 ? GetLimits().GetNumCols() : GetLimits().GetNumRows();
    MatrixT ret(D,D);
    Fill(ret,std::complex<double>(0.0));

    if (GetLimits().GetNumRows()==1)
    {
        //Left boundary

        for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
            ret+=conj(Transpose(*ip))*(*ip);
    }
    if (GetLimits().GetNumCols()==1)
    {
        //Right boundary
        for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
            ret+=(*ip)*conj(Transpose(*ip));
    }

    //cout << "OverlapTransferMatrix=" << ret << endl;
    return ret;
}

//
//  N1(n;i,l)=Sum[j,E(a-1,i,j)*A(n;j,l)
//  N2(n;k,l)=Sum(i,A^t(n;i,k)*N1(n;i,l])
//  E(a,k,l)=Sum{n,N2(n;k,l)}
//
MatrixProductSite::MatrixT MatrixProductSite::GetOverlapTransferMatrix(const MatrixT& Em) const
{
    int D=itsAs[0].GetNumCols();
    MatrixT Ea(D,D);
    Fill(Ea,std::complex<double>(0.0));

    pVectorT N1s;
    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
        N1s.push_back(Em*(*ip));

    cpIterT iN1=N1s.begin();
    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++,iN1++)
        Ea+=conj(Transpose(*ip))*(*iN1);

    return Ea;
}

MatrixProductSite::MatrixT MatrixProductSite::
GetOverlapMatrix(const MatrixT& Eleft, const MatrixT Eright) const
{
    MatrixT Sab(GetLimits());
    for (int ia=0;ia<itsp;ia++)
    {
        Sab+=Eleft*itsAs[ia]*Transpose(Eright);
    }
    return Sab;
}
