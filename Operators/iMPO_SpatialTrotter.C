#include "iMPO_SpatialTrotter.H"
#include "TensorNetworks/iHamiltonian.H"
#include "TensorNetworksImp/Typedefs.H"
#include "Containers/Matrix4.H"
#include "Operators/SiteOperatorImp.H"
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
iMPO_SpatialTrotter::iMPO_SpatialTrotter(double dt, Trotter type,const iHamiltonian* H)
    : iMPO_SpatialTrotter(dt,type,H,H->GetL())
{

}
iMPO_SpatialTrotter::iMPO_SpatialTrotter(double dt, Trotter type, const iHamiltonian* H, int L)
    : iMPOImp(L,H->GetS())
{
    int d=Getd();
    assert(type==Odd || type==Even);

    Matrix4RT exph=H->GetExponentH(dt);
    int N=exph.Flatten().GetNumRows();
//
// Now SVD to factor exp(-dt*H)
//
    LapackSVDSolver<double> solver;
    auto [U,sm,VT]=solver.Solve(exph.Flatten(),1e-12,N); //Solves A=U * s * VT

    //
    //  Load up site operators with special ops at the edges
    //
    switch (type)
    {
    case Even :  // LRLRLR  or LRLRLRI
        {
            if (L%2)
            { //Odd # of sites
               Insert(new SiteOperatorImp(d,expH));
               for (int ia=2;ia<=L;ia++)
               {
                    Insert(new SiteOperatorImp(d,DLeft,PBulk ,U ,sm));
                    ia++;
                    if (ia!=L)
                        Insert(new SiteOperatorImp(d,DRight,PBulk  ,VT,sm));
                    else
                        Insert(new SiteOperatorImp(d,DRight,PBulk  ,VT,sm));
               }
            }
            else
            { //Even # of sites
               for (int ia=1;ia<=L;ia+=2)
               {
                    Insert(new SiteOperatorImp(d,DRight,PBulk  ,VT,sm));
                    Insert(new SiteOperatorImp(d,DLeft ,PBulk ,U ,sm));
               }
            }
            break;
        }
    case Odd : // RLRLRL or RLRLRLRI
        {
            if (L%2)
            { //Odd # of sites
               Insert(new SiteOperatorImp(d,DLeft,PBulk  ,U ,sm));
               for (int ia=2;ia<L;ia++)
               {
                    Insert(new SiteOperatorImp(d,DRight ,PBulk ,VT,sm));
                    ia++;
                    if (ia==L)
                        Insert(new SiteOperatorImp(d,expH));
                    else
                        Insert(new SiteOperatorImp(d,DLeft,PBulk  ,U ,sm));
               }
            }
            else
            { //Even # of sites
                Insert(new SiteOperatorImp(d,DLeft,PBulk ,U ,sm));
                for (int ia=2;ia<L;ia+=2)
                {
                    Insert(new SiteOperatorImp(d,DRight,PBulk  ,VT,sm));
                    Insert(new SiteOperatorImp(d,DLeft ,PBulk ,U ,sm));
                }
                Insert(new SiteOperatorImp(d,DRight ,PBulk ,VT,sm));
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

iMPO_SpatialTrotter::~iMPO_SpatialTrotter()
{
    //dtor
}

}
