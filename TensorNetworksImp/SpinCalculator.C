#include "SpinCalculator.H"
#include <cmath>

SpinCalculator::SpinCalculator(double S)
    : itsS(S)
    , itsd(2*S+1)
{
#ifdef DEBUG
    double ipart;
    double frac=std::modf(2.0*itsS,&ipart);
    assert(frac==0.0);
#endif
    //ctor
}

SpinCalculator::~SpinCalculator()
{
    //dtor
}

double SpinCalculator::GetSm (int m, int n) const
{
    assert(m>=0);
    assert(n>=0);
    double sm=ConvertToSpin(m);
    double sn=ConvertToSpin(n);
    double ret=0.0;
    if (m+1==n)
    {
        ret=sqrt(itsS*(itsS+1.0)-sm*sn);
    }

    return ret;
}

double SpinCalculator::GetSp (int m, int n) const
{
    assert(m>=0);
    assert(n>=0);
    double sm=ConvertToSpin(m);
    double sn=ConvertToSpin(n);
    double ret=0.0;
    if (m==n+1)
    {
        ret=sqrt(itsS*(itsS+1.0)-sm*sn);
    }

    return ret;
}
double SpinCalculator::GetSx (int m, int n) const
{
    return 0.5*(GetSm(m,n)+GetSp(m,n));
}

SpinCalculator::complx SpinCalculator::GetSy (int m, int n) const
{
    return complx(0.0,0.5*(GetSm(m,n)-GetSp(m,n)));
}
double SpinCalculator::GetSz (int m, int n) const
{
    assert(m>=0);
    assert(n>=0);
    double sm=ConvertToSpin(m);
    double ret=0.0;
    if (m==n)
    {
        ret=sm;
    }

    return ret;
}

template <class M,class F> M SpinCalculator::BuildMatrix(int lowerIndex,F fp) const
{
    M ret(lowerIndex,itsd,lowerIndex,itsd);
    for (int m=lowerIndex;m<=itsd-1+lowerIndex;m++)
        for (int n=lowerIndex;n<=itsd-1+lowerIndex;n++)
            ret(m,n)=(this->*fp)(m-lowerIndex,n-lowerIndex);
    return ret;
}
template <class M,class F1, class F2> M SpinCalculator::BuildMatrix(int lowerIndex,F1 fp1,F2 fp2) const
{
    M ret(itsd,itsd,itsd,itsd,lowerIndex);
    for (int m1=lowerIndex;m1<=itsd-1+lowerIndex;m1++)
        for (int n1=lowerIndex;n1<=itsd-1+lowerIndex;n1++)
            for (int m2=lowerIndex;m2<=itsd-1+lowerIndex;m2++)
                for (int n2=lowerIndex;n2<=itsd-1+lowerIndex;n2++)
                    ret(m1,m2,n1,n2)=(this->*fp1)(m1-lowerIndex,n1-lowerIndex)*(this->*fp2)(m2-lowerIndex,n2-lowerIndex);
    return ret;
}

SpinCalculator::MatrixRT  SpinCalculator::GetSm (int lowerIndex) const
{
    return BuildMatrix<MatrixRT,dfp>(lowerIndex,&SpinCalculator::GetSm);
}
SpinCalculator::MatrixRT  SpinCalculator::GetSp (int lowerIndex) const
{
    return BuildMatrix<MatrixRT,dfp>(lowerIndex,&SpinCalculator::GetSp);
}
SpinCalculator::MatrixRT  SpinCalculator::GetSx (int lowerIndex) const
{
    return BuildMatrix<MatrixRT,dfp>(lowerIndex,&SpinCalculator::GetSx);
}
SpinCalculator::MatrixCT SpinCalculator::GetSy (int lowerIndex) const
{
    return BuildMatrix<MatrixCT,cfp>(lowerIndex,&SpinCalculator::GetSy);
}
SpinCalculator::MatrixRT  SpinCalculator::GetSz (int lowerIndex) const
{
    return BuildMatrix<MatrixRT,dfp>(lowerIndex,&SpinCalculator::GetSz);
}

SpinCalculator::Matrix4T  SpinCalculator::GetSxSx (int lowerIndex) const
{
    return BuildMatrix<Matrix4T,dfp,dfp>(lowerIndex,&SpinCalculator::GetSx,&SpinCalculator::GetSx);
}
SpinCalculator::Matrix4CT SpinCalculator::GetSxSy (int lowerIndex) const
{
    return BuildMatrix<Matrix4CT,dfp,cfp>(lowerIndex,&SpinCalculator::GetSx,&SpinCalculator::GetSy);
}
SpinCalculator::Matrix4T  SpinCalculator::GetSxSz (int lowerIndex) const
{
    return BuildMatrix<Matrix4T,dfp,dfp>(lowerIndex,&SpinCalculator::GetSx,&SpinCalculator::GetSz);
}
SpinCalculator::Matrix4CT SpinCalculator::GetSySx (int lowerIndex) const
{
    return BuildMatrix<Matrix4CT,cfp,dfp>(lowerIndex,&SpinCalculator::GetSy,&SpinCalculator::GetSx);
}
SpinCalculator::Matrix4CT SpinCalculator::GetSySy (int lowerIndex) const
{
    return BuildMatrix<Matrix4CT,cfp,cfp>(lowerIndex,&SpinCalculator::GetSy,&SpinCalculator::GetSy);
}
SpinCalculator::Matrix4CT SpinCalculator::GetSySz (int lowerIndex) const
{
    return BuildMatrix<Matrix4CT,cfp,dfp>(lowerIndex,&SpinCalculator::GetSy,&SpinCalculator::GetSz);
}
SpinCalculator::Matrix4T  SpinCalculator::GetSzSx (int lowerIndex) const
{
    return BuildMatrix<Matrix4T,dfp,dfp>(lowerIndex,&SpinCalculator::GetSz,&SpinCalculator::GetSx);
}
SpinCalculator::Matrix4CT SpinCalculator::GetSzSy (int lowerIndex) const
{
    return BuildMatrix<Matrix4CT,dfp,cfp>(lowerIndex,&SpinCalculator::GetSz,&SpinCalculator::GetSy);
}
SpinCalculator::Matrix4T  SpinCalculator::GetSzSz (int lowerIndex) const
{
    return BuildMatrix<Matrix4T,dfp,dfp>(lowerIndex,&SpinCalculator::GetSz,&SpinCalculator::GetSz);
}
