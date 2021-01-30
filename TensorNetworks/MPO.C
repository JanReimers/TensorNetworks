#include "TensorNetworks/MPO.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworks/SVCompressor.H"
#include "oml/matrix.h"

namespace TensorNetworks
{

const SiteOperator* MPO::GetSiteOperator(int isite) const
{
    return const_cast<MPO*>(this)->GetSiteOperator(isite);
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

std::string MPO::GetNormStatus () const
{
     int L=GetL();
     std::string status(L,' ');
     for (int ia=1;ia<=L;ia++)
           status[ia-1]=GetSiteOperator(ia)->GetNormStatus(1e-13);
     return status;
}



void MPO::Product(const MPO* O2)
{
    int L=GetL();
    assert(L==O2->GetL());
    for (int ia=1;ia<=L;ia++)
        GetSiteOperator(ia)->Product(O2->GetSiteOperator(ia));
}

double MPO::Compress(CompressType ct,int Dmax, double epsSV)
{
    Factory* f=Factory::GetFactory();
    SVCompressorR* comp=f->MakeMPOCompressor(Dmax,epsSV);
    double compressionError=Compress(ct,comp);
    delete comp;
    delete f;
    return compressionError;
}

double MPO::Compress(CompressType ct, const SVCompressorR* compressor)
{
    double terror=0.0;
    int L=GetL();
    for (int ia=1;ia<L;ia++)
    {
        double err=GetSiteOperator(ia)->Compress(ct,DLeft ,compressor);
        terror+=err*err;
    }
    for (int ia=L;ia>1;ia--)
    {
        double err=GetSiteOperator(ia)->Compress(ct,DRight ,compressor);
        terror+=err*err;
    }
    return sqrt(terror)/(L-1); //Truncation error per site.
}

void MPO::CanonicalForm()
{
    int L=GetL();
    for (int ia=1;ia<L;ia++)
        GetSiteOperator(ia)->CanonicalForm(DLeft);
    for (int ia=L;ia>1;ia--)
        GetSiteOperator(ia)->CanonicalForm(DRight);
}


}
