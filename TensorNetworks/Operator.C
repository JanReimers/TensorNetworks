#include "TensorNetworks/Operator.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"

namespace TensorNetworks
{

int Operator::GetMaxDw() const
{
    int L=GetL();
    int Dw=0;
    for (int ia=1;ia<L;ia++)
    {
        Dw=Max(Dw,GetSiteOperator(ia)->GetDw12().Dw2);
    }
    return Dw;
}

void Operator::Report(std::ostream& os) const
{
    int L=GetL();
    os << "Matrix Product Operator for " << L << " sites." << std::endl;
    for (int ia=1;ia<=L;ia++)
    {
        os << "   Site " << ia << ": ";
        GetSiteOperator(ia)->Report(os);
        os << std::endl;
    }
}

const SiteOperator* Operator::GetSiteOperator(int isite) const
{
    return const_cast<Operator*>(this)->GetSiteOperator(isite);
}


}
