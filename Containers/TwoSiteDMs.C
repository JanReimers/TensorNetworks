#include "TwoSiteDMs.H"
#include "oml/cnumeric.h"


TwoSiteDMs::TwoSiteDMs(int L, int p)
    : itsL(L)
    , itsp(p)
    , itsDMs(0,L-1,0,L-1)
{
    //ctor
}

TwoSiteDMs::~TwoSiteDMs()
{
    //dtor
}

void TwoSiteDMs::Insert(int ia, int ib,const DMType& dm)
{
    assert(dm.Flatten().GetNumRows()==itsp*itsp);
    assert(dm.Flatten().GetNumCols()==itsp*itsp);
    assert(dm.IsHermitian(1e-14));
    itsDMs(ia,ib)=dm;
}

template <class O> TwoSiteDMs::ExpectationT TwoSiteDMs::Contract(const O& op) const
{
    assert(op.Flatten().GetNumRows()==itsp*itsp);
    assert(op.Flatten().GetNumCols()==itsp*itsp);
//    assert(op.IsHermitian(1e-14));
//    cout << "op=" << op << endl;
    ExpectationT ret(0,itsL-1,0,itsL-1);
    Fill(ret,0.0);
    for (int ia=0; ia<itsL-1;ia++)
        for (int ib=ia+1; ib<itsL; ib++)
        {
            eType ex(0.0);
            const DMType& dm=itsDMs(ia,ib);
//            cout << "dm=" << dm << endl;
            for (int m1=1; m1<=itsp; m1++)
                for (int n1=1; n1<=itsp; n1++)
                    for (int m2=1; m2<=itsp; m2++)
                        for (int n2=1; n2<=itsp; n2++)
                            ex+=op(m1,m2,n1,n2)*dm(m1,m2,n1,n2);
            assert(fabs(std::imag(ex))<1e-14);
//            cout << "ia,ib,ex=" << ia << " " << ib << " " << ex << endl;
            ret(ia,ib)=std::real(ex);
        }
    return ret;
}
template TwoSiteDMs::ExpectationT TwoSiteDMs::Contract<TwoSiteDMs::OperatorT >(const OperatorT & op) const;
template TwoSiteDMs::ExpectationT TwoSiteDMs::Contract<TwoSiteDMs::OperatorCT>(const OperatorCT& op) const;

TwoSiteDMs::ExpectationT TwoSiteDMs::GetTraces() const
{
    ExpectationT ret(0,itsL-1,0,itsL-1);
    Fill(ret,0.0);
    for (int ia=0; ia<itsL-1;ia++)
        for (int ib=ia+1; ib<itsL; ib++)
            ret(ia,ib)=std::real(Sum(itsDMs(ia,ib).Flatten().GetDiagonal()));
    return ret;
}

double VNEntropy(const Vector<double>& s)
{
    static double eps=1e-14;
    int N=s.size();
    double ret=0.0;
    for (int i=1;i<=N;i++)
    {
        assert(s(i)>=-eps);
        if (s(i)>1.0) cout << "S(i)-1=" << s(i)-1 << endl;
        assert(s(i)<=1.0+1000*eps);  //This check is really tough to satify
        if (s(i)>0.0) ret+=s(i)*std::log(s(i));
    }
    return -ret/std::log(N);
}

TwoSiteDMs::ExpectationT TwoSiteDMs::GetVNEntropies() const
{
    ExpectationT ret(0,itsL-1,0,itsL-1);
    Fill(ret,0.0);
    for (int ia=0; ia<itsL-1;ia++)
        for (int ib=ia+1; ib<itsL; ib++)
        {
            //Vector<double> s=EigenValuesOnly<double,MatrixCT>(itsDMs(ia,ib).Flatten());
            Vector<eType> s(itsDMs(ia,ib).Flatten().GetDiagonal());
//            cout << "s(" << ia << "," << ib << ")=" << s << ", sum=" << Sum(s) << endl;
            ret(ia,ib)=VNEntropy(real(s));
        }
    return ret;
}

#define TYPE TwoSiteDMs::DMType
#include "oml/src/smatrix.cc"
