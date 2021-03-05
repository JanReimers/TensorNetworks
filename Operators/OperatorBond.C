#include "Operators/OperatorBond.H"
#include "Operators/SiteOperatorImp.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Typedefs.H"
#include "oml/diagonalmatrix.h"
#include "oml/matrix.h"
#include <iostream>
#include <iomanip>

namespace TensorNetworks
{

OperatorBond::OperatorBond(int D)
    : itsSingularValues(D)
    , itsEpsSV(0.0)
    , itsBondEntropy(0.0)
    , itsMinSV(0.0)
    , itsMaxDelta(0.0)
    , itsTruncationError(0.0)
    , itsD(D)
    , itsRank(0)
    , itsLeft_Site(0)
    , itsRightSite(0)
{
    assert(D>0);
    Fill(itsSingularValues,1.0/D);  //Start with maximum entanglement.
}

OperatorBond::~OperatorBond()
{
    //dtor
}

void OperatorBond::SetSites(SiteOperator* left, SiteOperator* right)
{
    itsLeft_Site=dynamic_cast<SiteOperatorImp*>(left );
    itsRightSite=dynamic_cast<SiteOperatorImp*>(right);
    assert(itsLeft_Site);
    assert(itsRightSite);
}

void OperatorBond::NewBondDimension(int D)
{
    assert(D>=1);
    assert(itsD==itsSingularValues.size());
    if (D>itsD)
    { //Grow
        double fill=0.0;
        itsSingularValues.SetLimits(D,true);

        for (int i=itsD+1;i<=D;i++)
            itsSingularValues(i)=fill;
        itsD=D;
        itsRank=D;
    }
    else
    { //Shrink/compress
        for (int i=D+1;i<=itsD;i++)
            itsTruncationError+=itsSingularValues(i)*itsSingularValues(i);
        itsSingularValues.SetLimits(D,true);
        itsMinSV=itsSingularValues(D);
        itsD=D;
        UpdateBondEntropy();
    }
}

void OperatorBond::SetSingularValues(const DiagonalMatrixRT& s,double compressionError)
{
    DiagonalMatrixRT s1=s;
    s1.ReBase(1);
    assert(itsD==itsSingularValues.GetNumRows());
    //std::cout << "SingularValues=" << s << std::endl;
    //DiagonalMatrixRT sn=s/GetNorm(s); //Make sure it is normalized
    if (s1.GetNumRows()!=itsD)
    { //Pad with zeros so we can get a delta l.
        int newD=s1.GetNumRows();
        itsSingularValues.SetLimits(newD,newD);
        for (int i=itsD+1;i<=newD;i++) itsSingularValues(i)=0.0;
    }
    itsMaxDelta=Max(fabs(s1-itsSingularValues));
//    std::cout << "new max delta =" << itsMaxDelta << " " << sn.GetDiagonal()-itsSingularValues.GetDiagonal() << std::endl;
    itsSingularValues=s1;
    itsD=s1.GetNumRows();

    itsRank=itsD;
    if (itsD>0)
        itsMinSV=itsSingularValues(itsD);
    itsTruncationError+=compressionError;
    UpdateBondEntropy();
}

void OperatorBond::ClearSingularValues(Direction lr,const MatrixRT& R)
{
    int D=0;
    switch (lr)
    {
    case DLeft:
        D=R.GetNumRows();
        break;
    case DRight:
        D=R.GetNumCols();
        break;
    default:
        assert(false);
    }
    if (D!=itsD)
    { //Pad with zeros so we can get a delta l.
        itsSingularValues.SetLimits(D,D);
    }
    itsMaxDelta=0.0;
    Fill(itsSingularValues,0.0);
    itsD=D;
    itsRank=itsD;
    itsMinSV=0.0;
    itsTruncationError=0.0;
    itsBondEntropy=0.0;
}


//double OperatorBond::GetNorm(const DiagonalMatrixRT& s)
//{
//    double s2=s.GetDiagonal()*s.GetDiagonal();
//    return sqrt(s2);
//}

void OperatorBond::UpdateBondEntropy()
{
    double norm_squared=itsSingularValues.GetDiagonal()*itsSingularValues.GetDiagonal();
    itsBondEntropy=0.0;
    for (int i=1;i<=itsD;i++)
    {
        double s2=(itsSingularValues(i)*itsSingularValues(i))/norm_squared;
        if (s2>0.0) itsBondEntropy-=s2*log(s2);
        if (fabs(itsSingularValues(i))<itsEpsSV)
        {
            //cout << "Auto rank reduction s=" << s << endl;
            itsRank--;
        }
    }
//    We dont normalize operator SVs so bond entropy can easily go negative, therefore no warning.
//    if (itsBondEntropy<0.0 && itsD>1)
//        std::cerr << "Warning negative bond entropy s=" << itsSingularValues << std::endl;
}

//
//  Direction is the normaliztions direction, which i opposite to the direction that UV gets tranferred.
//
void OperatorBond::GaugeTransfer(Direction lr,double compressionError,const DiagonalMatrixRT& s,const MatrixRT& R)
{
    SetSingularValues(s,compressionError);
    assert(GetSite(lr));
    // We have to forward the un-normalized Svs, otherwise R/L normalization breaks.
    GetSite(lr)->QLTransfer(lr,R);
}

void OperatorBond::GaugeTransfer(Direction lr,const MatrixRT& R)
{
    ClearSingularValues(lr,R);
    assert(GetSite(lr));
    // We have to forward the un-normalized Svs, otherwise R/L normalization breaks.
    GetSite(lr)->QLTransfer(lr,R);
}

void OperatorBond::CanonicalTransfer(Direction lr,double compressionError,const DiagonalMatrixRT& s,const MatrixRT& UV)
{
    SetSingularValues(s,compressionError);
    assert(GetSite(lr));
//    GetSite(lr)->SVDTransfer(lr,UV);
}

void OperatorBond::Report(std::ostream& os) const
{
    os
                                                  << std::setw(4)  << itsD
                                                  << std::setw(4)  << itsRank
       << std::fixed      << std::setprecision(6) << std::setw(12) << itsBondEntropy
       << std::scientific << std::setprecision(1) << std::setw(10) << itsMinSV
       << std::scientific << std::setprecision(1) << std::setw(10) << sqrt(itsTruncationError)
       ;
       if (itsMinSV>0.0 && itsD<20)
           os << " " << itsSingularValues.GetDiagonal();

}

// double OperatorBond::GetMaxDelta(const OperatorBond& cache) const
// {
//     VectorRT s=cache.itsSingularValues.GetDiagonal();
//     int Dold=s.size();
//     s.SetLimits(itsD,true);
//     for (int i=Dold+1;i<=itsD;i++) s(i)=0.0;
//     assert(itsSingularValues.size()==s.size());
//     return Max(fabs(itsSingularValues.GetDiagonal()-s));
// }

} //namespace
