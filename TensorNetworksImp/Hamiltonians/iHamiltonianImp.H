#ifndef iHAMILTONIANIMP_H
#define iHAMILTONIANIMP_H

#include "TensorNetworks/iHamiltonian.H"
#include "Operators/iMPOImp.H"
#include "Containers/Matrix4.H"

namespace TensorNetworks
{
//
//  This class has been wittled down to the point there all it does now is:
//      1) Cache H12 matrix
//      2) Act as factory for wave functions and other MPOs.
//
class iHamiltonianImp
    : public virtual iHamiltonian
    , private iMPOImp
{
public:
    iHamiltonianImp(int L, const OperatorClient* W);
    virtual ~iHamiltonianImp();

    virtual int    GetL    () const {return iMPOImp::GetL    ();}
    virtual double GetS    () const {return iMPOImp::GetS    ();}
    virtual int    Getd    () const {return iMPOImp::Getd    ();}
    virtual int    GetMaxDw() const {return iMPOImp::GetMaxDw();}

    virtual iTEBDState* CreateiTEBDState   (      int D,iTEBDType,double normEps,double epsSV) const;
    virtual iTEBDState* CreateiTEBDState   (int L,int D,iTEBDType,double normEps,double epsSV) const;
    virtual iMPO*       CreateiUnitOperator() const;
    virtual iMPO*       CreateiMPO         (double dt, TrotterOrder,CompressType,double epsMPO) const;
    virtual Matrix4RT   GetLocalMatrix     () const {return itsH12;}

private:
    Matrix4RT itsH12;
};

} //namespace

#endif // iHAMILTONIANIMP_H
