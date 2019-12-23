#include "MatrixProductSite.H"
#include <complex>

MatrixProductSite::MatrixProductSite(int p, int D1, int D2)
    : itsp(p)
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

void MatrixProductSite::InitializeWithProductState()
{
    for (pIterT ip=itsAs.begin();ip!=itsAs.end();ip++)
        (*ip)(1,1)=std::complex<double>(1.0);
}

//#include <iostream>
MatrixProductSite::MatrixT MatrixProductSite::GetOverlapTransferMatrix() const
{
    int nra=itsAs[0].GetNumRows();
    int nca=itsAs[0].GetNumCols();
    MatrixT ret(nra*nra,nca*nca);
    Fill(ret,std::complex<double>(0.0));

    for (int ia=1; ia<=nra; ia++)
        for (int ja=1; ja<=nca; ja++)
            for (int ib=1; ib<=nra; ib++)
                for (int jb=1; jb<=nca; jb++)
                    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
                    {
//                        std::cout << "A* A=" << conj((*ip)(ia,ja)) <<  " " << (*ip)(ib,jb) << std::endl;
//                        std::cout << "ia+nra*(ib-1),ja+nca*(jb-1))=" << ia+nra*(ib-1) << " " << ja+nca*(jb-1) << std::endl;
                        ret(ia+nra*(ib-1),ja+nca*(jb-1))+=conj((*ip)(ia,ja))*(*ip)(ib,jb);
                    }

    return ret;
}

MatrixProductSite::MatrixT MatrixProductSite::
GetOverlapMatrix(const MatrixT& Eleft, const MatrixT Eright) const
{
    MatrixT Sab(itsp,itsp);
    for (int ia=1;ia<=itsp;ia++)
        for (int ib=1;ib<=itsp;ib++)
        {
            Sab(ia,ib)=GetOverlapMatrixElement(itsAs[ia-1],itsAs[ib-1],Eleft,Eright);
        }
    return Sab;
}

std::complex<double> MatrixProductSite::
GetOverlapMatrixElement(const MatrixT& Aa, const MatrixT& Ab, const MatrixT& Eleft, const MatrixT& Eright) const
{
    assert(Aa.GetLimits()==Ab.GetLimits());
    int nra=Aa.GetNumRows();
    int nca=Aa.GetNumCols();
    MatrixT ret(nra*nra,nca*nca);
    Fill(ret,std::complex<double>(0.0));

    for (int ia=1; ia<=nra; ia++)
        for (int ja=1; ja<=nca; ja++)
            for (int ib=1; ib<=nra; ib++)
                for (int jb=1; jb<=nca; jb++)
                {
//                        std::cout << "A* A=" << conj((*ip)(ia,ja)) <<  " " << (*ip)(ib,jb) << std::endl;
//                        std::cout << "ia+nra*(ib-1),ja+nca*(jb-1))=" << ia+nra*(ib-1) << " " << ja+nca*(jb-1) << std::endl;
                    ret(ia+nra*(ib-1),ja+nca*(jb-1))+=conj(Aa(ia,ja))*Ab(ib,jb);
                }

    MatrixT Sab=Eleft*ret*Eright;
    assert(Sab.GetNumRows()==1);
    assert(Sab.GetNumCols()==1);
    return Sab(1,1);
}
