#include "TensorNetworks/MPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworksImp/SVMPOCompressor.H"
#include "oml/vector.h"
#include <cassert>

namespace TensorNetworks
{

void MPO::Combine(const Operator* O2)
{
    int L=GetL();
    assert(L==O2->GetL());
    for (int ia=1;ia<=L;ia++)
    {
        GetSiteOperator(ia)->Combine(O2->GetSiteOperator(ia));
    }
}

double MPO::Compress(int Dmax, double minSv)
{
    int L=GetL();
    SVMPOCompressor* compressor=new SVMPOCompressor(Dmax,minSv);
    Vector<int> oldDws(L),newDws(L);
    for (int ia=1;ia<L;ia++)
    {
        oldDws(ia)=GetSiteOperator(ia)->GetDw12().Dw2;
        GetSiteOperator(ia)->Compress(DLeft ,compressor);
    }
    oldDws(L)=0;
    for (int ia=L;ia>1;ia--)
    {
        GetSiteOperator(ia)->Compress(DRight,compressor);
        newDws(ia)=GetSiteOperator(ia)->GetDw12().Dw1;
    }
    newDws(1)=0;
    double percent=100-(100.0*Sum(newDws))/static_cast<double>(Sum(oldDws));
//    cout << "% compression=" << std::fixed << std::setprecision(2) << percent << endl;
    delete compressor;
    return percent;
}

int MPO::GetMaxDw() const
{
    int L=GetL();
    int Dw=0;
    for (int ia=1;ia<L;ia++)
    {
        Dw=Max(Dw,GetSiteOperator(ia)->GetDw12().Dw2);
    }
    return Dw;
}


void MPO::Report(std::ostream& os) const
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

}
