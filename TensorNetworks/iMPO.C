#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "oml/matrix.h"
#include <cassert>

namespace TensorNetworks
{
double iMPO::Compress(CompressType ct, const SVCompressorR* compressor)
{
    double terror=0.0;
    int L=GetL();
    if (ct==Parker)
    {
        CanonicalFormQRIter(DRight);
        for (int ia=1;ia<=L;ia++)
            GetSiteOperator(ia)->ZeroRowCol(DLeft); //Zero out row or column outside the V block
        CanonicalFormQRIter(DLeft); //Store gauge transforms for each site
        for (int ia=1;ia<=L;ia++)
        {
    //        std::cout << "Site " << ia << ": ";
            terror+=GetSiteOperator(ia)->iCompress(ct,DRight ,compressor); //Works on the gauge transforms we just stored.
        }

    }
    else
    {
        for (int ia=1;ia<=L;ia++)
            terror+=GetSiteOperator(ia)->iCompress(ct,DLeft  ,compressor); //Works on the gauge transforms we just stored.
        for (int ia=L;ia>=1;ia--)
            terror+=GetSiteOperator(ia)->iCompress(ct,DRight ,compressor);
    }


    return sqrt(terror)/(L-1); //Truncation error per site.
}

void iMPO::CanonicalForm()
{
//    GetSiteOperator(1)->iCanonicalFormQRIter(DLeft);
    GetSiteOperator(1)->iCanonicalFormTriangular(DLeft);
//    int L=GetL();
//    for (int ia=1;ia<=L;ia++)
//        GetSiteOperator(ia)->iCanonicalForm(DLeft);
//    for (int ia=L;ia>=1;ia--)
//        GetSiteOperator(ia)->iCanonicalForm(DRight);
}

void iMPO::CanonicalFormTri()
{
    GetSiteOperator(1)->iCanonicalFormTriangular(DLeft);
}

void iMPO::CanonicalFormQRIter(Direction lr)
{
    int L=GetL();
    for (int ia=1;ia<=L;ia++)
        GetSiteOperator(ia)->InitQRIter(); //Reset all G's to unit

    double eps=1e-13; //Cutoff for RR QR
    double eta=0.0;
    int niter=0,maxIter=20;
    do
    {
        eta=0.0;
        switch (lr)
        {
        case DLeft:
            for (int ia=1;ia<=L;ia++)
                eta=Max(eta,GetSiteOperator(ia)->QRStep(lr,eps));
            break;
        case DRight:
            for (int ia=L;ia>=1;ia--)
                eta=Max(eta,GetSiteOperator(ia)->QRStep(lr,eps));
            break;
        }
        niter++;
    } while (eta>1e-13 && niter <maxIter);
    if (niter==maxIter)
        std::cout << "CanonicalFormQRIter failed to converge, eta=" << eta << std::endl;
}


} //namespace
