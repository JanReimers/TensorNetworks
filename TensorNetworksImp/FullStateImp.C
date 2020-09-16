#include "FullStateImp.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworksImp/StateIterator.H"
#include "NumericalMethods/PrimeEigenSolver.H"
#include "Containers/SparseMatrix.H"
#include "oml/random.h"
#include "oml/array_io.h"


FullStateImp::FullStateImp(int L, double S)
    : itsL(L)
    , itsS(S)
    , itsE(0)
{
    assert(itsL>0);
#ifdef DEBUG
    double ipart;
    double frac=std::modf(2.0*itsS,&ipart);
    assert(frac==0.0);
#endif
    assert(itsS>=0.5);
    itsd=2*itsS+1;
    assert(itsd>1);
    itsN=static_cast<long int>(std::pow(itsd,itsL));
    assert(itsN>0);
    itsAmplitudes .SetSize(itsN);
    FillRandom(itsAmplitudes);
    //ctor
}

FullStateImp::~FullStateImp()
{
    //dtor
}

std::ostream& FullStateImp::Dump(std::ostream& os) const
{
    os.setf(std::ios::floatfield,std::ios::fixed);
    os.precision(9);

    os << "Full wave function L=" << itsL << " S=" << itsS << " N=" << itsN << " E=" << itsE << endl;
    os.precision(5);

    for (StateIterator is(itsL,itsd); !is.end(); is++)
    {
        TensorNetworks::eType a=itsAmplitudes[is.GetLinearIndex()];
        double anorm=sqrt(real(conj(a)*a));
        if (anorm>1e-8)
        {
            cout << "|";
            for (int ia=1; ia<=itsL; ia++)
            {
                cout << is.GetQuantumNumbers()(ia);
                if (ia<itsL) cout << ",";
            }
            cout << ">  " << anorm << " " << a << endl;
        }
    }
    os << endl;
    return os;
}





double FullStateImp::Contract(const Matrix4T& Hlocal)
{
    assert(Hlocal.Flatten().GetNumRows()==itsd*itsd);
    assert(Hlocal.Flatten().GetNumCols()==itsd*itsd);

    ArrayCT newAmplitudes(itsAmplitudes.size());
    Fill(newAmplitudes,TensorNetworks::eType(0.0));
    for (int ia=1; ia<=itsL-1; ia++)
    {
        ContractLocal(ia,Hlocal,newAmplitudes,itsAmplitudes);
    }

    // Evaluate E=<psi|H|psi>
    TensorNetworks::eType Ec=Dot(conj(itsAmplitudes),newAmplitudes);
    assert(fabs(imag(Ec))<1e-14);
    itsE=real(Ec);
    // Scale out the eigen value
    newAmplitudes/=itsE;
    // Find the change in psi
    double deltaPsi=Max(abs(newAmplitudes-itsAmplitudes));
    // Assign and normalize the updated amplitudes.
    itsAmplitudes=newAmplitudes;
    Normalize(itsAmplitudes);
    return deltaPsi;
}

FullStateImp::ArrayCT FullStateImp::Contract(const Matrix4T& Hlocal,const ArrayCT& oldAmpliudes) const
{
    assert(Hlocal.Flatten().GetNumRows()==itsd*itsd);
    assert(Hlocal.Flatten().GetNumCols()==itsd*itsd);

    ArrayCT newAmplitudes(itsAmplitudes.size());
    Fill(newAmplitudes,TensorNetworks::eType(0.0));
    for (int ia=1; ia<=itsL-1; ia++)
    {
        ContractLocal(ia,Hlocal,newAmplitudes,oldAmpliudes);
    }

    return newAmplitudes;
}


void FullStateImp::ContractLocal(int isite, const Matrix4T& Hlocal, ArrayCT& newAmplitudes, const ArrayCT& oldAmpliudes) const
{
    assert(isite>0);
    assert(isite<=itsL);
    for (StateIterator is(itsL,itsd); !is.end(); is++)
    {
        const Vector<int>& QNs=is.GetQuantumNumbers();
        int na=QNs(isite);
        int nb=QNs(isite+1);
        Vector<int> mstate=QNs;
        TensorNetworks::eType c(0.0);
        for (int ma=0; ma<itsd; ma++)
            for (int mb=0; mb<itsd; mb++)
            {
                mstate(isite  )=ma;
                mstate(isite+1)=mb;
                c+=Hlocal(ma,mb,na,nb)*oldAmpliudes[is.GetIndex(mstate)];
            }
//        cout << "stateVector=" << stateVector << endl;
//        cout << "indexn+1,GetIndex=" << indexn+1 << " " << GetIndex(stateVector) << endl;
        assert(is.GetLinearIndex()==is.GetIndex(QNs));
        newAmplitudes[is.GetLinearIndex()]+=c;

    }
}

void FullStateImp::Normalize(ArrayCT& amplitudes)
{
    TensorNetworks::eType E=Dot(conj(amplitudes),amplitudes);
    assert(real(E)>0.0);
    assert(fabs(imag(E))<1e-14);
    amplitudes/=sqrt(E);
    TensorNetworks::eType phase(1.0,0.0);
    for (int i=1; i<itsN; i++)
    {
        double r=std::fabs(amplitudes[i]);
        if (r>0.01)
        {
            phase=conj(amplitudes[i])/r;
            break;
        }
    }
    amplitudes*=phase; //Try and take out arbitrary phase factor.
}



double FullStateImp::PowerIterate(const IterationScheduleLine& sched,const Hamiltonian& H,bool quite)
{
    Normalize(itsAmplitudes);
    double E=0;
    Matrix4T Hlocal=H.BuildLocalMatrix();
//    os.setf(std::ios::floatfield,std::ios::fixed);
    if (!quite)
    {
        cout.unsetf(std::ios_base::floatfield);
        cout << "Iter#  dE    deltaPsi" << endl;
        cout << "---------------------" << endl;
    }

    for (int n=1; n<sched.itsMaxGSSweepIterations; n++)
    {
        double deltaPsi=Contract(Hlocal);
        double dE=fabs(itsE-E);
        E=itsE;
        if (fabs(deltaPsi)<sched.itsEps.itsEigenSolverEpsilon
                &&            dE<sched.itsEps.itsDelatEnergy1Epsilon) break;
        if (!quite)
            cout << n << " " << dE << " " << deltaPsi << endl;
    }
    return E;
}

double FullStateImp::FindGroundState(const IterationScheduleLine& sched,const Hamiltonian& H,bool quite)
{
    Matrix4T Hlocal=H.BuildLocalMatrix();
    double E=0;
    int Neig=1;
    PrimeEigenSolver<eType> solver;
    solver.Solve(Hlocal,this,Neig,sched.itsEps);
    itsE=solver.GetEigenValues()(1);
    const TensorNetworks::VectorCT amplitudes=solver.GetEigenVector(1);
    for (int i=1;i<=itsN;i++) itsAmplitudes[i-1]=amplitudes(i);
    Normalize(itsAmplitudes);
    return E;
}

//
//  Called from the Lanczos M*v routine. Do yvec=H*xvec; where H=sum{Hlocal(ia,ia+1}
//  This is a const member function so we don't use the member amplitudes.
//
void FullStateImp::DoHContraction (int N, eType* xvec, eType* yvec, const Matrix4T& Hlocal) const
{
    assert(N==itsN);
    ArrayCT oldAmplitudes(itsN);
    for (int i=0;i<itsN;i++) oldAmplitudes[i]=xvec[i];
    ArrayCT newAmplitudes=Contract(Hlocal,oldAmplitudes);
    for (int i=0;i<itsN;i++) yvec[i]=newAmplitudes[i];
}

