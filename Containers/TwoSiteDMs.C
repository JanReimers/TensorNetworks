#include "Containers/TwoSiteDMs.H"
#include "TensorNetworks/VNEntropy.H"
#include "oml/cnumeric.h"


TwoSiteDMs::TwoSiteDMs(int L, int d)
    : itsL(L)
    , itsd(d)
    , itsDMs(1,L,1,L)
{
    //ctor
}

TwoSiteDMs::~TwoSiteDMs()
{
    //dtor
}

void TwoSiteDMs::Insert(int ia, int ib,const DMType& dm)
{
    assert(dm.Flatten().GetNumRows()==itsd*itsd);
    assert(dm.Flatten().GetNumCols()==itsd*itsd);
    assert(dm.IsHermitian(1e-14));
    itsDMs(ia,ib)=dm;
}

template <class O> TwoSiteDMs::ExpectationT TwoSiteDMs::Contract(const O& op) const
{
    assert(op.Flatten().GetNumRows()==itsd*itsd);
    assert(op.Flatten().GetNumCols()==itsd*itsd);
//    assert(op.IsHermitian(1e-14));
//    cout << "op=" << op << endl;
    ExpectationT ret(itsL,itsL);
    Fill(ret,0.0);
    for (int ia=1; ia<=itsL-1;ia++)
        for (int ib=ia+1; ib<=itsL; ib++)
        {
            dcmplx ex(0.0);
            const DMType& dm=itsDMs(ia,ib);
//            cout << "dm=" << dm << endl;
            for (int m1=1; m1<=itsd; m1++)
                for (int n1=1; n1<=itsd; n1++)
                    for (int m2=1; m2<=itsd; m2++)
                        for (int n2=1; n2<=itsd; n2++)
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
    ExpectationT ret(itsL,itsL);
    Fill(ret,0.0);
    for (int ia=1; ia<=itsL-1;ia++)
        for (int ib=ia+1; ib<=itsL; ib++)
            ret(ia,ib)=std::real(Sum(itsDMs(ia,ib).Flatten().GetDiagonal()));
    return ret;
}


TwoSiteDMs::ExpectationT TwoSiteDMs::GetVNEntropies() const
{
    ExpectationT ret(itsL,itsL);
    Fill(ret,0.0);
    for (int ia=1; ia<=itsL-1;ia++)
        for (int ib=ia+1; ib<=itsL; ib++)
        {
            //Vector<double> s=EigenValuesOnly<double,MatrixCT>(itsDMs(ia,ib).Flatten());
            Vector<dcmplx> s(itsDMs(ia,ib).Flatten().GetDiagonal());
//            cout << "s(" << ia << "," << ib << ")=" << s << ", sum=" << Sum(s) << endl;
            ret(ia,ib)=VNEntropyFromEVs(real(s));
        }
    return ret;
}

#define TYPE TwoSiteDMs::DMType
#include "oml/src/smatrix.cpp"
