#include "TensorNetworks/MPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/SVCompressor.H"
#include "oml/vector.h"
#include "oml/matrix.h"
#include <cassert>

namespace TensorNetworks
{

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

void  MPO::Dump(std::ostream& os) const
{
    int L=GetL();
    os << "Matrix Product Operator for " << L << " sites." << std::endl;
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
const SiteOperator* MPO::GetSiteOperator(int isite) const
{
    return const_cast<MPO*>(this)->GetSiteOperator(isite);
}


void MPO::Combine(const MPO* O2)
{
    Combine(O2,1.0);
}

void MPO::Combine(const MPO* O2,double factor)
{
    int L=GetL();
    assert(L==O2->GetL());
    for (int ia=1;ia<=L;ia++)
    {
        GetSiteOperator(ia)->Combine(O2->GetSiteOperator(ia),factor);
    }
}

double MPO::Compress(int Dmax, double epsSV)
{
    Factory* f=Factory::GetFactory();
    SVCompressorR* comp=f->MakeMPOCompressor(Dmax,epsSV);
    double compressionError=Compress(comp);
    delete comp;
    delete f;
    return compressionError;
}


double MPO::Compress(const SVCompressorR* compressor)
{
//    GetSiteOperator(2)->Compress(DLeft ,compressor);
//    return 0.0;
    int L=GetL();
    Vector<int> oldDws(L),newDws(L);
    for (int ia=2;ia<L;ia++)
    {
        oldDws(ia)=GetSiteOperator(ia)->GetDw12().Dw2;
        GetSiteOperator(ia)->Compress(DLeft ,compressor);
    }
//    oldDws(L)=0;
//    for (int ia=L;ia>1;ia--)
//    {
//        GetSiteOperator(ia)->Compress(DRight,compressor);
//        newDws(ia)=GetSiteOperator(ia)->GetDw12().Dw1;
//    }
//    newDws(1)=0;
//    double percent=100-(100.0*Sum(newDws))/static_cast<double>(Sum(oldDws));
////    std::cout << "% compression=" << std::fixed << std::setprecision(2) << percent << std::endl;
    return 0.0;
}

void MPO::CanonicalForm(Direction lr)
{
//    GetSiteOperator(3)->CanonicalForm(lr);
    int L=GetL();
    switch (lr)
    {
        case DLeft:
            for (int ia=2;ia<L;ia++)
                GetSiteOperator(ia)->CanonicalForm(lr);
            break;
        case DRight:
            for (int ia=L-1;ia>1;ia--)
                GetSiteOperator(ia)->CanonicalForm(lr);
            break;
    }
}

std::string MPO::GetNormStatus () const
{
     int L=GetL();
     std::string status(L,' ');
     for (int ia=1;ia<=L;ia++)
           status[ia-1]=GetSiteOperator(ia)->GetNormStatus(1e-13);
     return status;
}


}
