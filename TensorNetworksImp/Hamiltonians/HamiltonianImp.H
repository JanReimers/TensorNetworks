#ifndef HAMILTONIANIMP_H
#define HAMILTONIANIMP_H

#include "TensorNetworks/Hamiltonian.H"
#include "Operators/MPOImp.H"
#include "Containers/Matrix4.H"

namespace TensorNetworks
{
//
//  Implement Hamiltonian construction which involves deciding what type of operators
//  (Left/Bulk/Right) live on each site.
//  This class has been wittled down to the point there all it does now is:
//      1) Cache H12 matrix
//      2) Act as factory for wave functions and other MPOs.
//
class HamiltonianImp
    : public virtual Hamiltonian
    , private MPOImp
{
public:
    HamiltonianImp(int L, const OperatorClient* W,TriType ul);
    virtual ~HamiltonianImp();

    virtual int    GetL    () const {return MPOImp::GetL    ();}
    virtual double GetS    () const {return MPOImp::GetS    ();}
    virtual int    Getd    () const {return MPOImp::Getd    ();}
    virtual int    GetMaxDw() const {return MPOImp::GetMaxDw();}

    virtual MPS*       CreateMPS          (int D          ,double normEps,double epsSV) const;
    virtual MPO*       CreateUnitOperator () const;
    virtual MPO*       CreateOperator     (double dt, TrotterOrder,CompressType,double epsMPO) const;
    virtual FullState* CreateFullState    () const;
    virtual Matrix4RT  GetLocalMatrix(         ) const {return itsH12;}

private:
    Matrix4RT itsH12;
};

} //namespace

#endif // HAMILTONIANIMP_H
