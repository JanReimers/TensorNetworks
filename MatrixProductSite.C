#include "MatrixProductSite.H"
#include <complex>

MatrixProductSite::MatrixProductSite(int p, int D1, int D2)
{
    for (int ip=0;ip<p;ip++)
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
MatrixProductSite::MatrixT MatrixProductSite::GetOverlap() const
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

