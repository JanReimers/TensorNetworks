#include "Operators/SiteOperatorImp.H"
#include "Operators/OperatorBond.H"
#include "TensorNetworks/SVCompressor.H"
#include "TensorNetworks/TNSLogger.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include "NumericalMethods/LapackQRSolver.H"
#include "NumericalMethods/LapackLinearSolver.H"
#include "oml/diagonalmatrix.h"

namespace TensorNetworks
{

using std::cout;
using std::endl;

double SiteOperatorImp::iCompress(CompressType ct,Direction lr,const SVCompressorR* comp)
{
    double terror=0.0;
    switch (ct)
    {
    case Std:
        terror=CompressStd(lr,comp);
        break;
    case Parker:
        terror=iCompressParker(lr,comp);
        break;
    case CNone:
        break;
    default:
        assert(false);
    }
    SetLimits();
    return terror;
}

void SiteOperatorImp::ZeroRowCol(Direction lr)
{
    auto [X1,X2]=itsWs.GetChi12();
    MatrixRT W0=itsWs.GetUnitTrace(); //<I,W>
//    cout << "W0=" << W0 << endl;
    VectorRT toprow=W0.GetRow(0);
    VectorRT c0=toprow.SubVector(1,X2);
//    cout << "c0=" << c0 << endl;
    MatrixRT A0=itsWs.ExtractM(W0,false);
//    cout << "A0=" << A0 << endl;
    MatrixRT Id(A0.GetLimits());
    Unit(Id);
    //
    //  Solver solves A*x=b, but here we need x*A=b, so we just transpose A to get that effect.
    //
    LinearSolver<double>* s=new LapackLinearSolver<double>();
    VectorRT t=s->SolveUpperTri(c0,Id-A0);
    MatrixRT G(W0.GetLimits());
    MatrixRT Ginv(W0.GetLimits());
    Unit(G);
    Unit(Ginv);
    for (int i=1;i<=X2;i++)
    {
        G(0,i)=t(i);
        Ginv(0,i)=-t(i);
    }
    assert(IsUnit(G*Ginv,1e-15));
#ifdef DEBUG
    W0=G*W0*Ginv;
    for (int i=1;i<=X2;i++)
        assert(fabs(W0(0,i))<1e-13);
#endif
    itsWs=G*itsWs*Ginv;
}

double SiteOperatorImp::iCompressParker(Direction lr,const SVCompressorR* comp)
{
    assert(comp);
    assert(lr==DRight);
    assert(GetNormStatus(1e-13)=='L');
    auto [X1,X2]=itsWs.GetChi12();
    for (int i=1;i<=itsG.GetNumCols()-2;i++)
        assert(fabs(itsG(0,i)<1e-13)); //Make sure top row got cleared out.
//    cout << "G=" << itsG << endl;
    MatrixRT C=itsWs.ExtractM(itsG,false);

    LapackSVDSolver<double> solver;
    auto [U,s,VT]=solver.SolveAll(C,1e-14);
//    cout << "s=" << s.GetDiagonal() << endl;
    double terror=comp->Compress(U,s,VT);
    int Xs=s.GetDiagonal().size();
//    cout << "U=" << U << endl;
    Grow(U,MatLimits(0,X1+1,0,Xs+1));
 //   cout << "U=" << U << endl;
    itsWs=Transpose(U)*itsWs*U; //We could also get a right version with VT*W*V
 //   cout << "orth=" << itsWs.GetOrthoMatrix(DLeft);
    SetLimits();
    return terror;
}



} //namespace
