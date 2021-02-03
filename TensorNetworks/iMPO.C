#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/SiteOperator.H"
#include <cassert>

namespace TensorNetworks
{
double iMPO::Compress(CompressType ct, const SVCompressorR* compressor)
{
    double terror=0.0;
    int L=GetL();
    for (int ia=1;ia<=L;ia++)
    {
//        std::cout << "Site " << ia << ": ";
        double err=GetSiteOperator(ia)->Compress(ct,DLeft ,compressor);
        terror+=err*err;
    }
    for (int ia=L;ia>=1;ia--)
    {
//        std::cout << "Site " << ia << ": ";
        double err=GetSiteOperator(ia)->Compress(ct,DRight ,compressor);
        terror+=err*err;
    }
    return sqrt(terror)/(L-1); //Truncation error per site.
}

void iMPO::CanonicalForm()
{
    int L=GetL();
        GetSiteOperator(1)->iCanonicalForm(DLeft);

//    for (int ia=1;ia<=L;ia++)
//        GetSiteOperator(ia)->iCanonicalForm(DLeft);
//    for (int ia=L;ia>=1;ia--)
//        GetSiteOperator(ia)->iCanonicalForm(DRight);
}

}
