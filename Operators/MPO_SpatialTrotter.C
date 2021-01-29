#include "MPO_SpatialTrotter.H"
#include "TensorNetworks/Hamiltonian.H"
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
MPO_SpatialTrotter::MPO_SpatialTrotter(double dt, Trotter type,int L, double S, const Hamiltonian* H)
    : MPOImp(L,S)
{
    assert(isValidSpin(S));
    int d=2*S+1;
    assert(L>1);
    assert(d>1);
    assert(type==Odd || type==Even);

    Matrix4RT expH=H->GetExponentH(dt);
    int N=expH.Flatten().GetNumRows();
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
            if (L%2)
            { //Odd # of sites
               Insert(new SiteOperatorLeft(d));
               for (int ia=2;ia<=L;ia++)
               {
                    Insert(new SiteOperatorBulk(d,DLeft ,U ,sm));
                    ia++;
                    if (ia!=L)
                        Insert(new SiteOperatorBulk(d,DRight ,VT,sm));
                    else
                        Insert(new SiteOperatorRight(d,DRight ,VT,sm));
               }
            }
            else
            { //Even # of sites
               Insert(new SiteOperatorLeft(d));
               for (int ia=2;ia<L;ia+=2)
               {
                    Insert(new SiteOperatorBulk(d,DLeft ,U ,sm));
                    Insert(new SiteOperatorBulk(d,DRight ,VT,sm));
               }
               Insert(new SiteOperatorRight(d));
            }
            break;
        }
    case Odd : // RLRLRL or RLRLRLRI
        {
            if (L%2)
            { //Odd # of sites
               Insert(new SiteOperatorLeft(d,DLeft ,U ,sm));
               for (int ia=2;ia<L;ia++)
               {
                    Insert(new SiteOperatorBulk(d,DRight ,VT,sm));
                    ia++;
                    if (ia==L)
                        Insert(new SiteOperatorRight(d));
                    else
                        Insert(new SiteOperatorBulk(d,DLeft ,U ,sm));
               }
            }
            else
            { //Even # of sites
                Insert(new SiteOperatorLeft(d,DLeft ,U ,sm));
                for (int ia=2;ia<L;ia+=2)
                {
                    Insert(new SiteOperatorBulk(d,DRight ,VT,sm));
                    Insert(new SiteOperatorBulk(d,DLeft ,U ,sm));
                }
                Insert(new SiteOperatorRight(d,DRight ,VT,sm));
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
