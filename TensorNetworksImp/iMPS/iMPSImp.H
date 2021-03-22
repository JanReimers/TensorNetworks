#ifndef IMATRIXPRODUCTSTATEIMP_H
#define IMATRIXPRODUCTSTATEIMP_H

#include "TensorNetworks/iMPS.H"
#include "TensorNetworks/Epsilons.H"
#include "TensorNetworksImp/iMPS/iMPSSite.H"
#include "Containers/ptr_vector.h"

namespace TensorNetworks
{

class iMPSImp : public virtual iMPS
{
//
//  Container typedefs
//
    typedef optr_vector<iMPSSite*> SitesType;
    typedef optr_vector<Bond*>     BondsType;

//--------------------------------------------------------------------------------------
//
//  Construction and initialization zone
//
public:
    iMPSImp(int L, double S, int D,double normEps,double epsSV);

    virtual ~iMPSImp();

    virtual void InitializeWith(State);
//--------------------------------------------------------------------------------------
//
//  Access to information
//
    virtual int     GetL        () const {return itsL;}
    virtual int     Getd        () const {return itsd;}
    virtual double  GetMaxDeltaE() const {return 0.0;}
    virtual double  GetMaxDeltal() const {return 0.0;} //Max delta lambda on bonds

    virtual void NormalizeSvD(Direction LR);
    virtual void NormalizeQR (Direction LR);
    virtual void Normalize();
//--------------------------------------------------------------------------------------
//
// Find ground state
//
    virtual double FindVariationalGroundState(const iHamiltonian* H,const IterationSchedule&);
//--------------------------------------------------------------------------------------
//
//    Overlap and expectation contractions
//
        virtual double   GetExpectation (const iMPO* o) const;
//        virtual dcmplx   GetExpectationC(const iMPO* o) const;
//--------------------------------------------------------------------------------------
//
//    Reporting
//
        virtual int          GetMaxD       (         ) const;
        virtual char         GetNormStatus (int isite) const;
        virtual std::string  GetNormStatus (         ) const;
        virtual void         Report    (std::ostream&) const;
        virtual bool         IsRLNormalized(int isite) const; //  AAA..AMBB..BBB form at one site.

//protected:
//    void IncreaseBondDimensions(int D);

//--------------------------------------------------------------------------------------
//
//  Allows unit test classes inside.
//
private:
    void InitSitesAndBonds(int D,double epsSV);

//--------------------------------------------------------------------------------------
//
//  For ground state search
//
        double FindVariationalGroundState(const iHamiltonian*,const IterationScheduleLine&);
//        void   Refine(Direction lr,const Hamiltonian*,const Epsilons&,int isite); //Minimize energy WRT the Ms in one site.
//        void   Sweep (Direction lr,const Hamiltonian*,const Epsilons&);
//        void   NormalizeSite             (Direction lr,int isite);


//--------------------------------------------------------------------------------------
//
//  Data members
//
    protected:
        int    itsL;   //Number of lattice sites
        double itsS;   //
        int    itsd;   //Number ot states for the spin on each site = 2S+1
        double itsNormEps;

        BondsType  itsBonds;
        SitesType  itsSites;

}; // class

} //namespace
#define SiteLoop(ia) for (int ia=1;ia<=itsL;ia++)
#define ForLoop(LR) for (int ia=GetStart(LR);ia!=GetStop(LR);ia+=GetIncrement(LR))

#define CheckSiteNumber(ia)\
    assert(ia>=1);\
    assert(ia<=itsL);\

#define CheckBondNumber(ib)\
    assert(ib>=1);\
    assert(ib<itsL);\

#endif // IMATRIXPRODUCTSTATEIMP_H
