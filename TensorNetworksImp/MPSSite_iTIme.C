#include "TensorNetworksImp/MPSSite.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
//#include "oml/minmax.h"
#include "oml/cnumeric.h"
//#include "oml/vector_io.h"
#include "oml/random.h"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

namespace TensorNetworks
{

void MPSSite::UpdateCache(const MPSSite* Psi2,const MatrixCT& Left, const MatrixCT& Right)
{
    itsLeft_Cache =IterateLeft_F(Psi2,Left);
    itsRightCache =IterateRightF(Psi2,Right);
}

// Get this site as close to psi as possible.  In the docs this site is psi^tilda
void MPSSite::Optimize(const MPSSite* psi, const MatrixCT& L, const MatrixCT& R)
{
    assert(itsd==psi->itsd);
    cout.precision(10);
    for (int n=0; n<itsd;n++)
    {
        MatrixCT Anew=ContractLRM(psi->itsMs[n],L,R);
//        cout << "A-Anew " << std::fixed << Max(abs(itsAs[n]-Anew)) << endl;
        itsMs[n]=Anew;
        }
}

} //namespace
