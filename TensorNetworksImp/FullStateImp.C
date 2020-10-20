#include "FullStateImp.H"
#include "TensorNetworks/Hamiltonian.H"
#include "TensorNetworks/IterationSchedule.H"
#include "TensorNetworksImp/StateIterator.H"
#include "TensorNetworks/CheckSpin.H"
#include "NumericalMethods/PrimeEigenSolver.H"
#include "Containers/SparseMatrix.H"
#include "Containers/Matrix4.H"
#include "oml/random.h"

namespace TensorNetworks
{

template <class T> FullStateImp<T>::FullStateImp(int L, double S)
    : itsL(L)
    , itsS(S)
    , itsd(2*S+1)
    , itsE(0)
{
    assert(itsL>0);
    assert(isValidSpin(S));
    assert(itsd>1);
    itsN=static_cast<long int>(std::pow(itsd,itsL));
    assert(itsN>0);
    itsAmplitudes.SetLimits(itsN);
    FillRandom(itsAmplitudes);
}

template <class T> FullStateImp<T>::~FullStateImp()
{
}

template <class T> std::ostream& FullStateImp<T>::Dump(std::ostream& os) const
{
    os.setf(std::ios::floatfield,std::ios::fixed);
    os.precision(9);

    os << "Full wave function L=" << itsL << " S=" << itsS << " N=" << itsN << " E=" << itsE << std::endl;
    os.precision(5);

    for (StateIterator is(itsL,itsd); !is.end(); is++)
    {
        dcmplx a=itsAmplitudes(is.GetLinearIndex());
        double anorm=sqrt(real(conj(a)*a));
        if (anorm>1e-8)
        {
            os << "|";
            for (int ia=1; ia<=itsL; ia++)
            {
                os << is.GetQuantumNumbers()(ia);
                if (ia<itsL) os << ",";
            }
            os << ">  " << anorm << " " << a << std::endl;
        }
    }
    os << std::endl;
    return os;
}

//
//  Fake out some complex functions so we can use the same code below for double and complex data types.
//
inline double conj(double& d) { return d;}
inline double real(double& d) { return d;}
inline double imag(double& d) { return 0.0;}

inline Vector<double>& conj(Vector<double>& v) { return v;}



template <class T> double FullStateImp<T>::OperateOverLattice()
{
    assert(itsHlocal.Flatten().GetNumRows()==itsd*itsd);
    assert(itsHlocal.Flatten().GetNumCols()==itsd*itsd);

    Vector<T> newAmplitudes=ContractOverLattice(itsAmplitudes);
    // Evaluate E=<psi|H|psi>
    T Ec=Dot(conj(itsAmplitudes),newAmplitudes);
    assert(fabs(imag(Ec))<1e-14);
    itsE=real(Ec);
    // Scale out the eigen value
    newAmplitudes/=itsE;
    // Find the change in psi
    double deltaPsi=Max(fabs(newAmplitudes-itsAmplitudes));
    // Assign and normalize the updated amplitudes.
    itsAmplitudes=newAmplitudes;
    Normalize(itsAmplitudes);
    return deltaPsi;
}

template <class T> Vector<T> FullStateImp<T>::ContractOverLattice(const Vector<T>& oldAmpliudes) const
{
    assert(itsHlocal.Flatten().GetNumRows()==itsd*itsd);
    assert(itsHlocal.Flatten().GetNumCols()==itsd*itsd);

    Vector<T> newAmplitudes(itsAmplitudes.size());
    Fill(newAmplitudes,T(0.0));
    for (int ia=1; ia<=itsL-1; ia++)
    {
        OperateLocal(oldAmpliudes,newAmplitudes,ia);
    }

    return newAmplitudes;
}


template <class T> void FullStateImp<T>::OperateLocal(const Vector<T>& oldAmpliudes, Vector<T>& newAmplitudes,int isite) const
{
    assert(itsHlocal.Flatten().GetNumRows()==itsd*itsd);
    assert(itsHlocal.Flatten().GetNumCols()==itsd*itsd);
    assert(isite>0);
    assert(isite<=itsL);
    for (StateIterator is(itsL,itsd); !is.end(); is++)
    {
        const Vector<int>& QNs=is.GetQuantumNumbers();
        int na=QNs(isite);
        int nb=QNs(isite+1);
        Vector<int> mstate=QNs;
        T c(0.0);
        for (int ma=0; ma<itsd; ma++)
            for (int mb=0; mb<itsd; mb++)
            {
                mstate(isite  )=ma;
                mstate(isite+1)=mb;
                c+=itsHlocal(ma,mb,na,nb)*oldAmpliudes(is.GetIndex(mstate));
            }
//        cout << "stateVector=" << stateVector << endl;
//        cout << "indexn+1,GetIndex=" << indexn+1 << " " << GetIndex(stateVector) << endl;
        assert(is.GetLinearIndex()==is.GetIndex(QNs));
        newAmplitudes(is.GetLinearIndex())+=c;

    }
}


template <class T> void FullStateImp<T>::Normalize(Vector<T>& amplitudes)
{
    double E=Dot(conj(amplitudes),amplitudes);
    assert(E>0.0);
    amplitudes/=sqrt(E);
    T phase(1.0);
    for (int i=1; i<=itsN; i++)
    {
        double r=std::fabs(amplitudes(i));
        if (r>0.01)
        {
            phase=conj(amplitudes(i))/r;
            break;
        }
    }
    amplitudes*=phase; //Try and take out arbitrary phase factor.
}



template <class T> double FullStateImp<T>::PowerIterate(const IterationScheduleLine& sched,const Hamiltonian& H,bool quite)
{
    Normalize(itsAmplitudes);
    double E=0;
    itsHlocal=H.BuildLocalMatrix();
    assert(itsHlocal.Flatten().GetNumRows()==itsd*itsd);
    assert(itsHlocal.Flatten().GetNumCols()==itsd*itsd);
//    os.setf(std::ios::floatfield,std::ios::fixed);
    if (!quite)
    {
        cout.unsetf(std::ios_base::floatfield);
        cout << "Iter#  dE    deltaPsi" << endl;
        cout << "---------------------" << endl;
    }

    for (int n=1; n<sched.itsMaxGSSweepIterations; n++)
    {
        double deltaPsi=OperateOverLattice();
        double dE=fabs(itsE-E);
        E=itsE;
        if (fabs(deltaPsi)<sched.itsEps.itsEigenSolverEpsilon
                &&            dE<sched.itsEps.itsDelatEnergy1Epsilon) break;
        if (!quite)
            cout << n << " " << dE << " " << deltaPsi << endl;
    }
    return E;
}

template <class T> double FullStateImp<T>::FindGroundState(const IterationScheduleLine& sched,const Hamiltonian& H,bool quite)
{
    itsHlocal=H.BuildLocalMatrix();
    assert(itsHlocal.Flatten().GetNumRows()==itsd*itsd);
    assert(itsHlocal.Flatten().GetNumCols()==itsd*itsd);
    double E=0;
    int Neig=1;
    PrimeEigenSolver<T> solver;
    solver.Solve1(this,Neig,sched.itsEps);
    itsE=solver.GetEigenValues()(1);
    VectorRT amp=solver.GetEigenVector(1);
    for (int i=1; i<=itsN; i++)
        itsAmplitudes(i)=amp(i);
    Normalize(itsAmplitudes);
    return E;
}

//
//  Called from the Lanczos M*v routine. Do yvec=H*xvec; where H=sum{Hlocal(ia,ia+1}
//  This is a const member function so we don't use the member amplitudes.
//
template <class T> void FullStateImp<T>::DoMatVecContraction (int N, T* xvec, T* yvec) const
{
    assert(itsHlocal.Flatten().GetNumRows()==itsd*itsd);
    assert(itsHlocal.Flatten().GetNumCols()==itsd*itsd);
    assert(N==itsN);
    Vector<T> oldAmplitudes(itsN); //TODO construct from (xvec,N) to avoid deep copy.
    for (int i=1;i<=itsN;i++) oldAmplitudes(i)=xvec[i-1];
    Vector<T> newAmplitudes=ContractOverLattice(oldAmplitudes);
    for (int i=1;i<=itsN;i++) yvec[i-1]=newAmplitudes(i);
}

template class FullStateImp<double>;
}

