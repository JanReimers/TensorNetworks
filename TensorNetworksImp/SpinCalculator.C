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

SpinCalculator::MatrixT  SpinCalculator::GetSm (int lowerIndex) const
{
    return BuildMatrix<MatrixT,dfp>(lowerIndex,&SpinCalculator::GetSm);
}
SpinCalculator::MatrixT  SpinCalculator::GetSp (int lowerIndex) const
{
    return BuildMatrix<MatrixT,dfp>(lowerIndex,&SpinCalculator::GetSp);
}
SpinCalculator::MatrixT  SpinCalculator::GetSx (int lowerIndex) const
{
    return BuildMatrix<MatrixT,dfp>(lowerIndex,&SpinCalculator::GetSx);
}
SpinCalculator::MatrixCT SpinCalculator::GetSy (int lowerIndex) const
{
    return BuildMatrix<MatrixCT,cfp>(lowerIndex,&SpinCalculator::GetSy);
}
SpinCalculator::MatrixT  SpinCalculator::GetSz (int lowerIndex) const
{
    return BuildMatrix<MatrixT,dfp>(lowerIndex,&SpinCalculator::GetSz);
}
