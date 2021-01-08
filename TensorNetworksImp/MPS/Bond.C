#include "TensorNetworksImp/MPS/Bond.H"
#include "TensorNetworksImp/MPS/MPSSite.H"
#include "oml/diagonalmatrix.h"
#include <iostream>
#include <iomanip>

namespace TensorNetworks
{

Bond::Bond(int D, double epsSV)
    : itsSingularValues(D)
    , itsEpsSV(epsSV)
    , itsBondEntropy(0.0)
    , itsMinSV(0.0)
    , itsMaxDelta(0.0)
    , itsCompessionError(-99)
    , itsD(D)
    , itsRank(0)
    , itsLeft_Site(0)
    , itsRightSite(0)
{
    assert(D>0);
    Fill(itsSingularValues,1.0/D);  //Start with maximum entanglement.
}

Bond::~Bond()
{
    //dtor
}

void Bond::CloneState(const Bond* b2)
{
    itsSingularValues =b2->itsSingularValues;
    itsEpsSV          =b2->itsEpsSV;
    itsBondEntropy    =b2->itsBondEntropy;
    itsCompessionError=b2->itsCompessionError;
    itsMinSV          =b2->itsMinSV;
    itsMaxDelta       =b2->itsMaxDelta;
    itsD              =b2->itsD;
    itsRank           =b2->itsRank;
}

Bond* Bond::Clone() const
{
    assert(this);
    Bond* b=new Bond(itsD,itsEpsSV);
    b->CloneState(this);
    return b;
}

void Bond::SetSites(MPSSite* left, MPSSite* right)
{
    itsLeft_Site=left;
    itsRightSite=right;
    assert(itsLeft_Site);
    assert(itsRightSite);
}

void Bond::NewBondDimension(int D)
{
    assert(D>=1);
    assert(itsD==itsSingularValues.size());
    if (D>itsD)
    { //Grow
        double fill=Min(itsSingularValues)/10.0;
        itsSingularValues.SetLimits(D,true);

        for (int i=itsD+1;i<=D;i++)
            itsSingularValues(i)=fill;
        itsD=D;
        itsRank=D;
    }
    else
    { //Shrink/compress
        itsCompessionError=0.0;
        for (int i=D+1;i<=itsD;i++)
            itsCompessionError+=itsSingularValues(i)*itsSingularValues(i);
        itsSingularValues.SetLimits(D,true);
        itsMinSV=itsSingularValues(D);
        itsD=D;
        UpdateBondEntropy();
    }
    itsSingularValues/=GetNorm(itsSingularValues);
}

void Bond::SetSingularValues(const DiagonalMatrixRT& s,double compressionError)
{
    assert(itsD==itsSingularValues.GetNumRows());
    //std::cout << "SingularValues=" << s << std::endl;
    DiagonalMatrixRT sn=s/GetNorm(s); //Make sure it is normalized
    if (sn.GetNumRows()!=itsD)
    { //Pad with zeros so we can get a delta l.
        int newD=sn.GetNumRows();
        itsSingularValues.SetLimits(newD,newD);
        for (int i=itsD+1;i<=newD;i++) itsSingularValues(i)=0.0;
    }
    itsMaxDelta=Max(fabs(sn-itsSingularValues));
//    std::cout << "new max delta =" << itsMaxDelta << " " << sn.GetDiagonal()-itsSingularValues.GetDiagonal() << std::endl;
    itsSingularValues=sn;
    itsD=s.GetNumRows();

    itsRank=itsD;
    itsMinSV=itsSingularValues(itsD);
    if (compressionError>0.0) itsCompessionError=compressionError;
    UpdateBondEntropy();
}

void Bond::ClearSingularValues(int D)
{
    if (D!=itsD)
    { //Pad with zeros so we can get a delta l.
        itsSingularValues.SetLimits(D,D);
    }
    itsMaxDelta=0.0;
    Fill(itsSingularValues,0.0);
    itsD=D;
    itsRank=itsD;
    itsMinSV=0.0;
    itsCompessionError=0.0;
    itsBondEntropy=0.0;
}


double Bond::GetNorm(const DiagonalMatrixRT& s)
{
    double s2=s.GetDiagonal()*s.GetDiagonal();
    return sqrt(s2);
}

void Bond::UpdateBondEntropy()
{
    itsBondEntropy=0.0;
    for (int i=1;i<=itsD;i++)
    {
        double s2=itsSingularValues(i)*itsSingularValues(i);
        if (s2>0.0) itsBondEntropy-=s2*log(s2);
        if (fabs(itsSingularValues(i))<itsEpsSV)
        {
            //cout << "Auto rank reduction s=" << s << endl;
            itsRank--;
        }
    }
    if (itsBondEntropy<0.0 && itsD>1)
        std::cerr << "Warning negative bond entropy s=" << itsSingularValues << std::endl;
}

//
//  Direction is the normaliztions direction, which i opposite to the direction that UV gets tranferred.
//
void Bond::SVDTransfer(Direction lr,double compressionError,const DiagonalMatrixRT& s,const MatrixCT& UV)
{
    SetSingularValues(s,compressionError);
    assert(GetSite(lr));
    // We have to forward the un-normalized Svs, otherwise R/L normalization breaks.
    GetSite(lr)->SVDTransfer(lr,s,UV);
}

void Bond::TransferQR(Direction lr,const MatrixCT& R)
{
    ClearSingularValues(R.GetNumRows());
    assert(GetSite(lr));
    // We have to forward the un-normalized Svs, otherwise R/L normalization breaks.
    GetSite(lr)->TransferQR(lr,R);
}

void Bond::CanonicalTransfer(Direction lr,double compressionError,const DiagonalMatrixRT& s,const MatrixCT& UV)
{
    SetSingularValues(s,compressionError);
    assert(GetSite(lr));
    GetSite(lr)->SVDTransfer(lr,UV);
}

void Bond::Report(std::ostream& os) const
{
    os
                                                  << std::setw(4)  << itsD
                                                  << std::setw(4)  << itsRank
       << std::fixed      << std::setprecision(6) << std::setw(12) << itsBondEntropy
       << std::scientific << std::setprecision(1) << std::setw(10) << itsMinSV
       << std::scientific << std::setprecision(1) << std::setw(10) << itsCompessionError
       ;

}

 double Bond::GetMaxDelta(const Bond& cache) const
 {
     VectorRT s=cache.itsSingularValues.GetDiagonal();
     int Dold=s.size();
     s.SetLimits(itsD,true);
     for (int i=Dold+1;i<=itsD;i++) s(i)=0.0;
     assert(itsSingularValues.size()==s.size());
     return Max(fabs(itsSingularValues.GetDiagonal()-s));
 }

} //namespace
