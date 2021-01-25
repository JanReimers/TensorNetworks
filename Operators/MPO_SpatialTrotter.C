#include "MPO_SpatialTrotter.H"
#include "TensorNetworksImp/Typedefs.H"
#include "Containers/Matrix4.H"
#include "Operators/SiteOperatorBulk.H"
#include "Operators/SiteOperatorLeft.H"
#include "Operators/SiteOperatorRight.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include "TensorNetworks/CheckSpin.H"
#include "oml/diagonalmatrix.h"
#include "oml/numeric.h"

namespace TensorNetworks
{

//
//  THe local two site Hamiltonian H12 should be stored as H12(m1,m2,n1,n2);
//   Whem flatted to H(m,n) where m=(m1,m2) n=(n1,n2) it is hermitian and diagonalizable.
//
MPO_SpatialTrotter::MPO_SpatialTrotter(double dt, Trotter type,int L, double S, const Matrix4RT& H12)
    : MPOImp(L,S,MPOImp::LoadLater)
{
    assert(isValidSpin(S));
    int d=2*S+1;
    assert(L>1);
    assert(d>1);
    assert(type==Odd || type==Even);
    //
    //  Diagonalize H12 in order to caluclate exp(-t*H)
    //
    //cout << "H12=" << H12 << endl;
    MatrixRT U12=H12.Flatten();
    VectorRT evs=Diagonalize(U12);
    assert(evs.size()==d*d);
    VectorRT expEvs=exp(-dt*evs);
    Matrix4RT expH(d,d,d,d,0);
    Fill(expH.Flatten(),0.0);
    int i1=1,N=U12.GetNumRows();
    for (int m1=0; m1<d; m1++)
        for (int m2=0; m2<d; m2++,i1++)
        {
            int i2=1;
            for (int n1=0; n1<d; n1++)
                for (int n2=0; n2<d; n2++,i2++)
                    for (int k=1; k<=N; k++)
                        expH(m1,n1,m2,n2)+=U12(i1,k)*expEvs(k)*U12(i2,k);
        }
//
// Now SVD to factor exp(-dt*H)
//
    LapackSVDSolver<double> solver;
    auto [U,sm,VT]=solver.Solve(expH.Flatten(),1e-12,N); //Solves A=U * s * VT

    //
    //  Load up site operators with special ops at the edges
    //
    switch (type)
    {
    case Even :  // LRLRLR  or LRLRLRI
        {
            for (int ia=1;ia<L;ia++)
            {
                if (ia==1)
                    Insert(new SiteOperatorLeft(d,DLeft ,U ,sm));
                else
                    Insert(new SiteOperatorBulk(d,DLeft ,U ,sm));

                ia++;

                if (ia==L)
                {
                    if (L%2)
                        Insert(new SiteOperatorBulk(d)); //if L is odd add one I op at the end
                    else
                        Insert(new SiteOperatorRight(d,DRight,VT,sm));
                }
                else
                    Insert(new SiteOperatorBulk(d,DRight,VT,sm));
            }
            break;
        }
    case Odd : // RLRLRL or RLRLRLRI
        {
            for (int ia=1;ia<=L-1;ia++)
            {
                if (ia==1)
                    Insert(new SiteOperatorLeft(d,DRight,VT,sm));
                else
                    Insert(new SiteOperatorBulk(d,DRight,VT,sm));
                ia++;
                if (ia==L)
                {
                    if (L%2)
                        Insert(new SiteOperatorRight(d)); //if L is odd add one I op at the end
                    else
                        Insert(new SiteOperatorRight(d,DLeft ,U ,sm));
                }
                if (ia<=L-1)
                    Insert(new SiteOperatorBulk(d,DLeft ,U ,sm));
            }
            break;
        }
    default :
        {
            assert(false);
        }
    }
    LinkSites();
}

MPO_SpatialTrotter::~MPO_SpatialTrotter()
{
    //dtor
}

}
