#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/SiteOperator.H"
#include <cassert>

namespace TensorNetworks
{

void iMPO::Product(const iMPO* O2)
{
    int L=GetL();
    assert(L==O2->GetL());
    for (int ia=1;ia<=L;ia++)
    {
        GetSiteOperator(ia)->Product(O2->GetSiteOperator(ia));
    }
}

}
