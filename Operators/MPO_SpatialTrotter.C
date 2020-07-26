#include "MPO_SpatialTrotter.H"
#include "TensorNetworksImp/Typedefs.H"
#include "Operators/SiteOperatorImp.H"
#include "Operators/IdentityOperator.H"
#include "oml/numeric.h"


typedef TensorNetworks::VectorT VectorT;
typedef TensorNetworks::MatrixT MatrixT;


MPO_SpatialTrotter::MPO_SpatialTrotter(double dt, TensorNetworks::Trotter type,int L, int p, const Matrix4T& Hlocal)
    : itsOddEven(type)
    , itsL(L)
    , itsp(p)
    , itsLeft_Site(0)
    , itsRightSite(0)
    , itsUnit_Site(0)
{
    assert(itsOddEven==TensorNetworks::Odd || itsOddEven==TensorNetworks::Even);
    assert(itsL>1);
    assert(itsp>1);
    //
    //  Diagonalize Hlocal
    //
    MatrixT U=Hlocal.Flatten();
    VectorT evs=Diagonalize(U);
    VectorT expEvs=exp(-dt*evs/2.0);
//    cout << "U=" << U << endl;
//    cout << "evs=" <<  evs << endl;
//    cout << "expEvs=" <<  expEvs << endl;
    //
    //  Now U is the matrix of eigen vectors
    //

    int Dw=evs.size();
    assert(Dw=itsp*itsp);
    OperatorWRepresentation* IdentityWOp=new IdentityOperator();

    itsLeft_Site=new SiteOperatorImp(TensorNetworks::DLeft ,U,expEvs   ,itsp);
    itsRightSite=new SiteOperatorImp(TensorNetworks::DRight,U,expEvs   ,itsp);
    itsUnit_Site=new SiteOperatorImp(TensorNetworks::PBulk ,IdentityWOp,itsp);

    delete IdentityWOp;
}

MPO_SpatialTrotter::~MPO_SpatialTrotter()
{
    //dtor
}


//
//  THis is non-trivial because we need return unit MPOs for special cases
//
const SiteOperator* MPO_SpatialTrotter::GetSiteOperator(int isite) const
{
    assert(isite>0);
    assert(isite<=itsL);
    const SiteOperator* ret=0;

    if (itsOddEven==TensorNetworks::Odd)
    {
        if (itsL%2 && isite==itsL)
        {  //odd number of sites
            ret=itsUnit_Site;
        }
        else
        {
            ret=(isite%2) ? itsLeft_Site : itsRightSite;
        }
    }
    else
    {
        if (isite==1 || (!(itsL%2) && isite==itsL))
        {  //odd number of sites
            ret=itsUnit_Site;
        }
        else
        {
            ret=(isite%2) ? itsRightSite : itsLeft_Site;
        }
    }
    return ret;
}
