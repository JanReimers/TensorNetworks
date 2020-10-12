#include "MPO_SpatialTrotter.H"
#include "TensorNetworksImp/Typedefs.H"
#include "Containers/Matrix4.H"
#include "Operators/SiteOperatorImp.H"
#include "Operators/IdentityOperator.H"
#include "NumericalMethods/LapackSVD.H"
#include "oml/diagonalmatrix.h"
#include "oml/numeric.h"

namespace TensorNetworks
{

//
//  THe local two site Hamiltonian H12 should be stored as H12(m1,m2,n1,n2);
//   Whem flatted to H(m,n) where m=(m1,m2) n=(n1,n2) it is hermitian and diagonalizable.
//
MPO_SpatialTrotter::MPO_SpatialTrotter(double dt, Trotter type,int L, int d, const Matrix4RT& H12)
    : itsOddEven(type)
    , itsL(L)
    , itsd(d)
    , itsLeft_Site(0)
    , itsRightSite(0)
    , itsUnit_Site(0)
{
    assert(itsOddEven==Odd || itsOddEven==Even);
    assert(itsL>1);
    assert(itsd>1);
    //
    //  Diagonalize H12 in order to caluclate exp(-t*H)
    //
    //cout << "H12=" << H12 << endl;
    MatrixRT U12=H12.Flatten();
    VectorRT evs=Diagonalize(U12);
    assert(evs.size()==itsd*itsd);
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
    auto [U,sm,VT]=LaSVDecomp(expH.Flatten()); //Solves A=U * s * VT
    OperatorWRepresentation* IdentityWOp=new IdentityOperator();

    itsLeft_Site=new SiteOperatorImp(DLeft,U,sm,itsd);
    itsRightSite=new SiteOperatorImp(DRight,VT,sm,itsd);
    itsUnit_Site=new SiteOperatorImp(PBulk,IdentityWOp,itsd);

    delete IdentityWOp;
}

MPO_SpatialTrotter::~MPO_SpatialTrotter()
{
    //dtor
}


//
//  THis is non-trivial because we need return unit MPOs for special cases
//
SiteOperator* MPO_SpatialTrotter::GetSiteOperator(int isite)
{
    assert(isite>0);
    assert(isite<=itsL);
    SiteOperator* ret=0;

    if (itsOddEven==Odd)
    {
        if (itsL%2 && isite==itsL)
        {
            //odd number of sites
            ret=itsUnit_Site;
        }
        else
        {
            // Odd site are left, even are right
            ret=(isite%2) ? itsLeft_Site : itsRightSite;
        }
    }
    else
    {
        if (isite==1 || (!(itsL%2) && isite==itsL))
        {
            //odd number of sites
            ret=itsUnit_Site;
        }
        else
        {
            //Odd sites are right, even are left.
            ret=(isite%2) ? itsRightSite : itsLeft_Site;
        }
    }
    /*
    cout << "Site " << isite << " is ";
    if (ret==itsLeft_Site) cout << "LEFT" << endl;
    if (ret==itsRightSite) cout << "RIGHT" << endl;
    if (ret==itsUnit_Site) cout << "UNIT" << endl;
    */
    return ret;
}

}
