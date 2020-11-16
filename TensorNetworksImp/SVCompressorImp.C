#include "SVCompressorImp.H"
#include "oml/diagonalmatrix.h"
#include "oml/matrix.h"
#include <iostream>

namespace TensorNetworks
{

template <class T> SVCompressorImp<T>::SVCompressorImp(int Dmax, double eps)
: itsDmax(Dmax)
, itsSVeps(eps)
{
}

template <class T> SVCompressorImp<T>::~SVCompressorImp()
{
}

template <class T> double SVCompressorImp<T>::Compress(MatrixT& U, DiagonalMatrixRT& s, MatrixT& Vdagger) const
{
    int N=s.GetNumRows();
    int D=N;
    double integratedS2=0.0;
    if (itsSVeps==0.0)
    {
        D=itsDmax;
        assert(D>0);
        for (int id=D+1;id<=N;id++)
            integratedS2+=s(id,id)*s(id,id);
    }
    else
    {
        //
        //  Scan backwards, smallest first to D such than sum(s^2,D..N) < eps
        //
        for (; D>=1; D--)
        {
            double s2=s(D,D)*s(D,D);
            if (integratedS2+s2>=itsSVeps) break;
            integratedS2+=s2;
        }
        if (itsDmax>0)
        {
            if (D>itsDmax)
            {
//                std::cerr << "Warning: SVCompressorImp::Compress loss of epslion control D=" << D << " > Dmax=" << itsDmax << std::endl;
                D=itsDmax;
            }
        }
    }
//
//  if required compress tensors and rescale SVs.
//
    if (D<N)
    {
        double S2BeforeTruncation=s.GetDiagonal()*s.GetDiagonal();
        assert(S2BeforeTruncation>0.0);
        //
        //  Trim tensors
        //
        s      .SetLimits(D,true);  // Resize s
        U      .SetLimits(U.GetNumRows(),D,true);
        Vdagger.SetLimits(D,Vdagger.GetNumCols(),true);

        double S2AfterTruncation=s.GetDiagonal()*s.GetDiagonal();
        assert(S2AfterTruncation>0.0);
        double rescaleS=S2BeforeTruncation/S2AfterTruncation;
        s*=rescaleS;
//        cout << "SVCompressorImptruncation D,N,intS2=" << D << " " << N << " " << integratedS2 << " " << s.GetLimits() << endl;
    }
    return sqrt(integratedS2);
}

template class SVCompressorImp<dcmplx>;
template class SVCompressorImp<double>;

}
