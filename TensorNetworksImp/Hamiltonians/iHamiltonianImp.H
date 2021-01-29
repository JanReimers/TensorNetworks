#ifndef iHAMILTONIANIMP_H
#define iHAMILTONIANIMP_H

#include "TensorNetworks/iHamiltonian.H"
#include "Operators/iMPOImp.H"
#include "Containers/Matrix4.H"

namespace TensorNetworks
{
//
//  Implement iHamiltonian construction with all Bulk operators (no Left/Right) at the edges.
//
class iHamiltonianImp
    : public virtual iHamiltonian
    , private iMPOImp
{
public:
    iHamiltonianImp(int L, const OperatorClient* W);
    virtual ~iHamiltonianImp();
    //
    //  iHamiltonian interface
    //
    virtual int GetL    () const {return iMPOImp::GetL    ();}
    virtual int Getd    () const {return iMPOImp::Getd    ();}
    virtual int GetMaxDw() const {return iMPOImp::GetMaxDw();}

    virtual iTEBDState* CreateiTEBDState   (int D,iTEBDType,double normEps,double epsSV) const;
    virtual iMPO*       CreateiUnitOperator() const;
    virtual iMPO*       CreateiMPO         (double dt, TrotterOrder, double epsMPO) const;
    virtual Matrix4RT   GetLocalMatrix     () const {return itsH12;}

private:
    Matrix4RT itsH12;
};

} //namespace

#endif // iHAMILTONIANIMP_H
