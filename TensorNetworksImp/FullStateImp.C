#include "FullStateImp.H"
#include "Containers/SparseMatrix.H"
#include "oml/random.h"
#include "oml/array_io.h"


FullStateImp::FullStateImp(int L, double S)
    : itsL(L)
    , itsS(S)
{
    assert(itsL>0);
    #ifdef DEBUG
    double ipart;
    double frac=std::modf(2.0*itsS,&ipart);
    assert(frac==0.0);
#endif
    assert(itsS>=0.5);
    itsp=2*itsS+1;
    assert(itsp>1);
    itsN=static_cast<long int>(std::pow(itsp,itsL));
    assert(itsN>0);
    itsAmplitudes .SetSize(itsN);
    itsAmplitudes1.SetSize(itsN);
    FillRandom(itsAmplitudes);
    //ctor
}

FullStateImp::~FullStateImp()
{
    //dtor
}

std::ostream& FullStateImp::Dump(std::ostream& os) const
{
    os << "Full wave function L=" << itsL << " S=" << itsS << " N=" << itsN << endl;
    os << "  Amplitudes=";
    for (int i=0;i<Min(1000L,itsN);i++) os << itsAmplitudes[i] << ", ";
    os << endl;
    return os;
}

// Caluclate the linear index from a rank L tensor index stored in state;
int FullStateImp::GetIndex(const Vector<int>& state) const
{
    int ret=0;
    for (int ia=1;ia<=itsL;ia++)
        ret=itsp*ret+state(ia);
    return ret; //0 based of Array type used for amplitudes
}

 void FullStateImp::Contract(const Matrix4T& Hlocal)
 {
    assert(Hlocal.Flatten().GetNumRows()==itsp*itsp);
    assert(Hlocal.Flatten().GetNumCols()==itsp*itsp);
    Vector<int> stateVector(itsL); //Used on contractions
    Fill(stateVector,0);
    for (int ia=1;ia<=itsL-1;ia++)
    {
        int indexn=0,indexm=0;
        ContractLocal(ia,1,stateVector,indexn,indexm,Hlocal);
    }
    itsAmplitudes=itsAmplitudes1;
 }

 void FullStateImp::ContractLocal(int isite,int ia,Vector<int>& stateVector,int& indexn,int& indexm, const Matrix4T& Hlocal)
 {
    assert(ia>0);
    assert(isite>0);
    assert(isite<=itsL);
    if (ia==isite) indexm=indexn;
    if (ia<=itsL)
    { //Keep recursing
        for (int na=0;na<itsp;na++)
        {
            stateVector(ia)=na;
            ContractLocal(isite,ia+1,stateVector,indexn,indexm,Hlocal);
        }
    }
    else
    {
        int na=stateVector(isite);
        int nb=stateVector(isite+1);
        Vector<int> mstate=stateVector;
        TensorNetworks::eType c(0.0);
        for (int ma=0;ma<itsp;ma++)
            for (int mb=0;mb<itsp;mb++)
            {
                mstate(isite  )=ma;
                mstate(isite+1)=mb;
                c+=Hlocal(ma,mb,na,nb)*itsAmplitudes[GetIndex(mstate)];
            }
//        cout << "stateVector=" << stateVector << endl;
//        cout << "indexn+1,GetIndex=" << indexn+1 << " " << GetIndex(stateVector) << endl;
        assert(indexn==GetIndex(stateVector));
        itsAmplitudes1[indexn]=c;
        indexn++;
    }

 }

 double FullStateImp::Normalize()
 {
    TensorNetworks::eType E=Dot(conj(itsAmplitudes),itsAmplitudes);
    assert(real(E)>0.0);
    assert(fabs(imag(E))<1e-14);
    itsAmplitudes/=sqrt(E);
    return real(E);
 }
