#include "TensorNetworks/iMPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/SVCompressor.H"
#include "oml/vector.h"
#include "oml/matrix.h"
#include <cassert>

namespace TensorNetworks
{

/*
int iMPO::GetMaxDw() const
{
    int L=GetL();
    int Dw=0;
    for (int ia=1;ia<L;ia++)
    {
        Dw=Max(Dw,GetSiteOperator(ia)->GetDw12().Dw2);
    }
    return Dw;
}
*/

/*void iMPO::Report(std::ostream& os) const
{
    int L=GetL();
    os << "infinite Matrix Product Operator for " << L << " sites." << std::endl;
    os << " Site    Dw      iDw      SVD      F-norm    Norm  U/L  LRB" << std::endl;
    os << "  #     1   2   1   2   Tr err    fin   inf  stat          " << std::endl;
    os << "-----------------------------------------------------------" << std::endl;
    for (int ia=1;ia<=L;ia++)
    {
        os << std::setw(4) << ia << ": ";
        GetSiteOperator(ia)->Report(os);
        os << std::endl;
    }
}

void  iMPO::Dump(std::ostream& os) const
{
    int L=GetL();
    os << "Infinite Matrix Product Operator for " << L << " sites." << std::endl;
    for (int ia=1;ia<=L;ia++)
    {
        os << "   Site " << ia << ": " << std::fixed << std::setprecision(3);
        const SiteOperator* so=GetSiteOperator(ia);
        int d=so->Getd();
        for (int m=0;m<d;m++)
        for (int n=0;n<d;n++)
            os << "W(" << m << "," << n << ")=" << so->GetW(m,n) << std::endl;
        os << std::endl;
    }
}

*/

void iMPO::Combine(const iMPO* O2)
{
    int L=GetL();
    assert(L==O2->GetL());
    for (int ia=1;ia<=L;ia++)
    {
        GetSiteOperator(ia)->Combine(O2->GetSiteOperator(ia),1.0);
    }
}
/*
double iMPO::CompressStd(int Dmax, double epsSV)
{
    Factory* f=Factory::GetFactory();
    SVCompressorR* comp=f->MakeMPOCompressor(Dmax,epsSV);
    double compressionError=CompressStd(comp);
    delete comp;
    delete f;
    return compressionError;
}


double iMPO::CompressStd(const SVCompressorR* compressor)
{
    int L=GetL();
    for (int ia=1;ia<L;ia++)
        GetSiteOperator(ia)->CompressStd(DLeft ,compressor);
    for (int ia=L;ia>1;ia--)
        GetSiteOperator(ia)->CompressStd(DRight,compressor);
    return 0.0;
}

double iMPO::CompressParker(int Dmax, double epsSV)
{
    Factory* f=Factory::GetFactory();
    SVCompressorR* comp=f->MakeMPOCompressor(Dmax,epsSV);
    double compressionError=CompressParker(comp);
    delete comp;
    delete f;
    return compressionError;
}


double iMPO::CompressParker(const SVCompressorR* compressor)
{
    int L=GetL();
    for (int ia=1;ia<L;ia++)
        GetSiteOperator(ia)->CompressParker(DLeft ,compressor);
    for (int ia=L;ia>1;ia--)
        GetSiteOperator(ia)->CompressParker(DRight,compressor);
    return 0.0;
}

void iMPO::CanonicalForm(Direction lr)
{
//    GetSiteOperator(3)->CanonicalForm(lr);
    int L=GetL();
    switch (lr)
    {
        case DLeft:
            for (int ia=1;ia<L;ia++)
                GetSiteOperator(ia)->CanonicalForm(lr);
            break;
        case DRight:
            for (int ia=L;ia>1;ia--)
                GetSiteOperator(ia)->CanonicalForm(lr);
            break;
    }
}

std::string iMPO::GetNormStatus () const
{
     int L=GetL();
     std::string status(L,' ');
     for (int ia=1;ia<=L;ia++)
           status[ia-1]=GetSiteOperator(ia)->GetNormStatus(1e-13);
     return status;
}
*/

}
