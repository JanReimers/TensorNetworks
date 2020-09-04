#include "TensorNetworksImp/MPSSite.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "oml/minmax.h"
#include "oml/cnumeric.h"
//#include "oml/vector_io.h"
#include "oml/random.h"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;


void MPSSite::Refine(const MatrixCT& Heff,const Epsilons& eps)
{
    assert(!isFrozen);
    assert(Heff.GetNumRows()==Heff.GetNumCols());
    int N=Heff.GetNumRows();
    Vector<double>  eigenValues(N);
    itsEigenSolver.Solve(Heff,2,eps); //Get lowest two eigen values/states

    eigenValues=itsEigenSolver.GetEigenValues();

    itsIterDE=eigenValues(1)-itsEmin;
    itsEmin=eigenValues(1);
    itsGapE=eigenValues(2)-eigenValues(1);
    Update(itsEigenSolver.GetEigenVector(1));
}

void MPSSite::Update(const VectorCT& newAs)
{
    Vector3<eType> As(itsd,itsD1,itsD2,newAs); //Unflatten
    for (int m=0; m<itsd; m++)
        for (int i1=1; i1<=itsD1; i1++)
            for (int i2=1; i2<=itsD2; i2++)
                itsMs[m](i1,i2)=As(m,i1,i2);

    itsNumUpdates++;
}

void MPSSite::UpdateCache(const SiteOperator* so, const Vector3T& HLeft, const Vector3T& HRight)
{
    itsHLeft_Cache=IterateLeft_F(so,HLeft);
    itsHRightCache=IterateRightF(so,HRight);
}
