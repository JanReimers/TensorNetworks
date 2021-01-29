#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/SiteOperator.H"
#include <cassert>

namespace TensorNetworks
{

void iMPO::Combine(const iMPO* O2)
{
    int L=GetL();
    assert(L==O2->GetL());
    for (int ia=1;ia<=L;ia++)
    {
        GetSiteOperator(ia)->Combine(O2->GetSiteOperator(ia),1.0);
    }
}

}
