#include "Containers/OneSiteDMs.H"
#include "TensorNetworks/VNEntropy.H"
#include "oml/cnumeric.h"


OneSiteDMs::OneSiteDMs(int L, int p)
    : itsL(L)
    , itsp(p)
    , itsDMs(L)
{
    //ctor
}

OneSiteDMs::~OneSiteDMs()
{
    //dtor
}

void OneSiteDMs::Insert(int ia, const DMType& dm)
{
    assert(dm.GetNumRows()==itsp);
    assert(dm.GetNumCols()==itsp);
    assert(IsHermitian(dm,1e-14));
    itsDMs[ia]=dm;
}

template <class O> OneSiteDMs::ExpectationT OneSiteDMs::Contract(const O& op) const
{
    assert(op.GetNumRows()==itsp);
    assert(op.GetNumCols()==itsp);
    assert(IsHermitian(op,1e-14));
//    cout << "op=" << op << endl;
    ExpectationT ret(itsL);
    Fill(ret,0.0);
    for (int ia=0; ia<itsL; ia++)
    {
        eType ex(0.0);
        const DMType& dm=itsDMs[ia];
//            cout << "dm=" << dm << endl;
        for (int m1=1; m1<=itsp; m1++)
            for (int n1=1; n1<=itsp; n1++)
                ex+=op(m1,n1)*dm(m1,n1);
        assert(fabs(std::imag(ex))<1e-14);
//            cout << "ia,ib,ex=" << ia << " " << ib << " " << ex << endl;
        ret[ia]=std::real(ex);
    }
    return ret;
}
template OneSiteDMs::ExpectationT OneSiteDMs::Contract<OneSiteDMs::OperatorT >(const OperatorT & op) const;
template OneSiteDMs::ExpectationT OneSiteDMs::Contract<OneSiteDMs::OperatorCT>(const OperatorCT& op) const;

OneSiteDMs::ExpectationT OneSiteDMs::GetTraces() const
{
    ExpectationT ret(itsL);
    Fill(ret,0.0);
    for (int ia=0; ia<itsL; ia++)
        ret[ia]=std::real(Sum(itsDMs[ia].GetDiagonal()));
    return ret;
}

OneSiteDMs::ExpectationT OneSiteDMs::GetVNEntropies() const
{
    ExpectationT ret(itsL);
    Fill(ret,0.0);
    for (int ia=0; ia<itsL; ia++)
    {
        //Vector<double> s=EigenValuesOnly<double,MatrixCT>(itsDMs(ia,ib).Flatten());
        Vector<eType> s(itsDMs[ia].GetDiagonal());
//            cout << "s(" << ia << "," << ib << ")=" << s << ", sum=" << Sum(s) << endl;
        ret[ia]=VNEntropyFromEVs(real(s));
    }
    return ret;
}

#define TYPE OneSiteDMs::DMType
#include "oml/src/smatrix.cc"
