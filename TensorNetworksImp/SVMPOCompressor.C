#include "SVMPOCompressor.H"
#include "oml/dmatrix.h"
#include "oml/diagonalmatrix.h"

namespace TensorNetworks
{

SVMPOCompressor::SVMPOCompressor(int Dmax, double eps)
: SVCompressorImp(Dmax,eps)
{
    //ctor
}

SVMPOCompressor::~SVMPOCompressor()
{
    //dtor
}

double SVMPOCompressor::Compress(MatrixT& U, DiagonalMatrixRT& sm, MatrixT& VT)
{
//    return SVCompressorImp::Compress(U,sm,VT);
    assert(itsDmax>=0);
    assert(itsSVeps>=0.0);
    assert(!(itsDmax==0 && itsSVeps==0.0));

    typedef VectorRT VectorRT;
    VectorRT s=sm.GetDiagonal();
    MatrixT A=U*sm*VT;
    int      M=A.GetNumRows(),N=A.GetNumCols();
    int      mn=Min(M,N);
    // At this point we have N singular values but we only Dmax of them or only the ones >=epsMin;
    int D=itsDmax>0 ? Min(mn,itsDmax) : mn; //Ignore Dmax if it is 0
    // Shrink so that all s(is<=D)>=epsMin;
    for (int is=D; is>=1; is--)
        if (s(is)>=itsSVeps)
        {
            D=is;
            break;
        }
    if (D<s.size())
    {
//        cout << "Smin=" << s(D) << "  Sum of rejected singular values=" << Sum(s.SubVector(D+1,s.size())) << endl;
//        cout << "S=" << s << endl;
    }
    double Sums=Sum(s);
    assert(Sums>0.0);
    s.SetLimits(D,true);  // Resize s
    sm.SetLimits(D,true);  // Resize s
    U.SetLimits(U.GetNumRows(),D,true);
    VT.SetLimits(D,VT.GetNumCols(),true);
    assert(Sum(s)>0.0);
    double rescaleS=Sums/Sum(s);
    sm*=rescaleS;
//    cout << "error=" << Max(abs(MatrixT(U*sm*VT-A))) << endl;
//    assert(Max(abs(MatrixT(U*sm*VT-A)))<10*itsSVeps);
    return 0.0;
}

} // namespace
