#include "MPO_SpatialTrotter.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworksImp/Typedefs.H"
#include "Containers/Matrix4.H"
#include "Operators/SiteOperatorImp.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include "TensorNetworks/TNSLogger.H"
#include "TensorNetworks/CheckSpin.H"
#include "oml/diagonalmatrix.h"
#include "oml/numeric.h"

namespace TensorNetworks
{

//
//  THe local two site Hamiltonian H12 should be stored as H12(m1,m2,n1,n2);
//   Whem flatted to H(m,n) where m=(m1,m2) n=(n1,n2) it is hermitian and diagonalizable.
//
MPO_SpatialTrotter::MPO_SpatialTrotter(double dt, Trotter type,const Hamiltonian* H)
    : MPOImp(H->GetL(),H->GetS())
{
    int d=Getd();
    int L=GetL();
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
//                Logger->LogWarn(0,"MPO_SpatialTrotter with odd number of lattice sites will not compress effectively");
                for (int ia=1;ia<=L-1;ia++)
                {
                    Insert(new SiteOperatorImp(d,DLeft,GetPosition(L,ia) ,U ,sm));
                    ia++;
                    Insert(new SiteOperatorImp(d,DRight,GetPosition(L,ia),VT,sm));
                }
                Insert(new SiteOperatorImp(d,expH)); //Identity op
            }
            else
            { //Even # of sites
                for (int ia=1;ia<=L;ia++)
                {
                    Insert(new SiteOperatorImp(d,DLeft ,GetPosition(L,ia),U ,sm));
                    ia++;
                    Insert(new SiteOperatorImp(d,DRight,GetPosition(L,ia),VT,sm));
                }
            }
            break;
        }
    case Odd : // RLRLRL or ILRLRLR
        {
            if (L%2)
            { //Odd # of sites
//               Logger->LogWarn(0,"MPO_SpatialTrotter with odd number of lattice sites will not compress effectively");
               Insert(new SiteOperatorImp(d,expH));
               for (int ia=2;ia<=L-1;ia++)
               {
                    Insert(new SiteOperatorImp(d,DLeft,GetPosition(L,ia) ,U,sm));
                    ia++;
                    Insert(new SiteOperatorImp(d,DRight,GetPosition(L,ia),VT ,sm));
               }
            }
            else
            { //Even # of sites
                for (int ia=1;ia<=L;ia++)
                {
                    Insert(new SiteOperatorImp(d,DRight,GetPosition(L,ia),VT,sm));
                    ia++;
                    Insert(new SiteOperatorImp(d,DLeft ,GetPosition(L,ia),U ,sm));
                }
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
