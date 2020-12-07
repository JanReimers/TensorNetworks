#include "Operators/iMPOImp.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "oml/vector.h"

namespace TensorNetworks
{

iMPOImp::iMPOImp(int L, double S,MPOImp::LoadWith loadWith)
    : MPOImp(L,S,loadWith)
{

}

iMPOImp::~iMPOImp()
{
    //dtor
}
//
//double iMPOImp::Compress(const SVCompressorR* compressor)
//{
//    int L=GetL();
//    Vector<int> oldDws(L),newDws(L);
//    double truncationError=0.0;
//    for (int ia=1;ia<=L;ia++)
//    {
//        oldDws(ia)=GetSiteOperator(ia)->GetDw12().Dw2;
//        GetSiteOperator(ia)->Compress(DLeft ,compressor);
//    }
//    oldDws(L)=0;
////    for (int ia=L;ia>=1;ia--)
////    {
////        GetSiteOperator(ia)->Compress(DRight,compressor);
////        newDws(ia)=GetSiteOperator(ia)->GetDw12().Dw1;
////    }
//    newDws(1)=0;
//    double percent=100-(100.0*Sum(newDws))/static_cast<double>(Sum(oldDws));
////    cout << "% compression=" << std::fixed << std::setprecision(2) << percent << endl;
//    return percent;
//}

} //namespace
