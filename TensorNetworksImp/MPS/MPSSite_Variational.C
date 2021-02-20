#include "TensorNetworksImp/MPS/MPSSite.H"
#include "TensorNetworksImp/MPS/Bond.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Epsilons.H"
#include "NumericalMethods/EigenSolver.H"

#include "oml/cnumeric.h"
#include "oml/random.h"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

namespace TensorNetworks
{

void MPSSite::Refine(const MatrixCT& Heff,const Epsilons& eps)
{
    assert(!isFrozen);
    assert(Heff.GetNumRows()==Heff.GetNumCols());
    int N=Heff.GetNumRows();
    //cout << "Heff N=" << N << endl;
    Vector<double>  eigenValues(N);
    auto [U,d]=itsEigenSolver->Solve(Heff,eps.itsEigenSolverEpsilon,2); //Get lowest two eigen values/states

    eigenValues=d;

    itsIterDE=d(1)-itsEmin;
    itsEmin=d(1);
    itsGapE=d(2)-(1);
    Update(U.GetColumn(1));
}

void MPSSite::Update(const VectorCT& newAs)
{
    Vector3<dcmplx> As(itsd,itsD1,itsD2,newAs); //Unflatten
    for (int m=0; m<itsd; m++)
        for (int i1=1; i1<=itsD1; i1++)
            for (int i2=1; i2<=itsD2; i2++)
                itsMs[m](i1,i2)=As(m,i1,i2);

    itsNumUpdates++;
}

void MPSSite::UpdateCache(const SiteOperator* so, const Vector3CT& HLeft, const Vector3CT& HRight)
{
    itsHLeft_Cache=IterateLeft_F(so,HLeft);
    itsHRightCache=IterateRightF(so,HRight);
}

} //namespace
