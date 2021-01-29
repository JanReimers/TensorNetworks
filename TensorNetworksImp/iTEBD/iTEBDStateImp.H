#ifndef ITEBDSTATEIMP_H
#define ITEBDSTATEIMP_H

#include "TensorNetworks/iTEBDState.H"
#include "TensorNetworks/Typedefs.H"
#include "TensorNetworksImp/MPS/MPSImp.H"
#include "oml/diagonalmatrix.h"

namespace TensorNetworks
{


//--------------------------------------------------------------------------------
//
//  Implement Vidals  ....Lambda-Gamma-Lambda-Gamma-Lambda... canonical form.
//  Try and re-use some of the MPSImp code through private inheritance
//  L in this class is interpreted as the number of Gammas in the (infinityl) repeating unit cell.
//
class iTEBDStateImp : public virtual iTEBDState,
    private MPSImp
{
public:
    using dVectorT=MPSSite::dVectorT;
//    typedef std::tuple<dVectorT,DiagonalMatrixRT> GLType;
    typedef std::tuple<MatrixCT,MatrixCT> MMType;
    typedef std::tuple<MatrixCT,double  > MdType;
    typedef std::vector<Matrix4RT>        MultigateType; // sequence of Trotter gates
//    typedef  ptr_vector< MPO*>            MultiMPOType; // sequence of Trotter gates as MPOs
    typedef  ptr_vector<iMPO*>            Multi_iMPOType; // sequence of Trotter gates as MPOs
public:
    iTEBDStateImp(int L,double S, int D,double normEps,double epsSV);
    virtual ~iTEBDStateImp();
    virtual void InitializeWith(State);
    virtual void ReCenter(int isite) const;
    virtual void IncreaseBondDimensions(int D);
    virtual void NewBondDimensions(int D1, int D2);
//--------------------------------------------------------------------------------------
//
//  Access to information
//
    virtual int GetL() const {return MPSImp::GetL();}
    virtual int Getp() const {return MPSImp::Getp();}
    virtual int GetD() const {return MPSImp::GetMaxD();}
//--------------------------------------------------------------------------------------
//
// Find ground state.  Return <E^2>-<E>^2
//
    virtual double FindiTimeGroundState      (const Hamiltonian* H,const iHamiltonian* iH,const IterationSchedule&);
    virtual void   InitGates (const iHamiltonian*,double dt,TrotterOrder,CompressType,double eps)=0;
    virtual void   Apply     (SVCompressorC*, int center)=0;

    virtual void   Apply     (const Matrix4RT&      expH,SVCompressorC* comp) {       Apply(expH,comp,false);}
    virtual void   Apply     (const MultigateType & expH,SVCompressorC* comp,int center);
//    virtual void   Apply     (const MultiMPOType  & expH,SVCompressorC* comp,int center);
    virtual void   Apply     (const Multi_iMPOType& expH,SVCompressorC* comp,int center);
    virtual double ApplyOrtho(const Matrix4RT&      expH,SVCompressorC* comp) {return Apply(expH,comp,true );}
    virtual double ApplyOrtho(const Matrix4RT&      expH,SVCompressorC*,double eps,int maxIter);

    virtual void   Apply     (const iMPO* expH,SVCompressorC* comp) {       Apply(expH,comp,false);}
    virtual double ApplyOrtho(const iMPO* expH,SVCompressorC* comp) {return Apply(expH,comp,true );}
    virtual double ApplyOrtho(const iMPO* expH,SVCompressorC*,double eps,int maxIter);

    double    Apply     (const Matrix4RT& expH,SVCompressorC*,bool orthogonalize);
    double    Apply     (const iMPO*      expH,SVCompressorC*,bool orthogonalize);
//--------------------------------------------------------------------------------------
//
//  Normalization members with and without SVD compression
//
    virtual void Canonicalize(Direction);  //Vidals L-G-L-G-L canonical form
    virtual void Normalize(Direction LR);

    virtual double Orthogonalize (SVCompressorC*);
    virtual double OrthogonalizeI(SVCompressorC*,double eps,int maxIter);

    virtual double GetExpectationDw1(const  MPO* o) const;
    virtual double GetExpectation   (const iMPO* o) const; //Recursive for triangular MPOs
    virtual double GetExpectationmmnn (const Matrix4RT& Hlocal) const;
    virtual double GetExpectationmnmn (const Matrix4RT& expH) const;
    double    GetExpectationDw1 (const  MPO*      o     ,int center) const;
    double    GetExpectation    (const iMPO*      o     ,int center) const;
    double    GetExpectationmmnn(const Matrix4RT& Hlocal,int center) const;

    virtual void Report(std::ostream&) const;
    virtual std::string GetNormStatus () const {return MPSImp::GetNormStatus();};

protected:

    int       GetModSite(int isite) const;
    double    FindiTimeGroundState(const Hamiltonian* H,const iHamiltonian* iH,const iMPO* H2, const IterationScheduleLine&);


    virtual   double GetOrthonormalityErrors() const;

    Matrix4CT GetTransferMatrix(Direction lr) const;

    enum ThetaType    {AlB,lAlB,lBlA,lAlBl,lBlAl,rBlAr};

    MatrixCT  ContractTheta(int ma, int mb,ThetaType tt) const;
    dVectorT  ContractAlB() const;
    dVectorT  ContractTheta(const Matrix4RT& expH,ThetaType) const;
    dVectorT  ContractTheta(const iMPO*       expH,ThetaType) const;
    dVectorT  ContractThetaDw(const iMPO*     expH) const;

    void     CalculateLambdaB (const dVectorT& thetaBA,SVCompressorC*);
    double   UnpackOrthonormal(const dVectorT& gammap,SVCompressorC*);
    void     Unpack_iMPO      (const dVectorT& gammap,SVCompressorC*);
    void     Unpack           (const dVectorT& gammap,SVCompressorC*);
//
//  All these helper function are static to ensure no dependence on iTEBD state
//
    static void      Orthogonalize (dVectorT& gamma, DiagonalMatrixRT& lambda);
    static void      OrthogonalizeI(dVectorT& gamma, DiagonalMatrixRT& lambda,double eps,int maxIter);
    static double    GetOrthonormalityErrors(const dVectorT& gamma, const DiagonalMatrixRT& lambda);
    static MdType    GetEigenMatrix(Direction lr, int D, const Matrix4CT& theta);
    static MMType    Factor(const MatrixCT m);
    static Matrix4CT GetTransferMatrix(const dVectorT& M); //M=lambda*gamma or gamma*lambda
    static MatrixCT  GetNormMatrix(Direction lr,const dVectorT& M); //Er*I or I*El
    static double    GetExpectation (const dVectorT& gamma,const DiagonalMatrixRT& lambda,const iMPO* o); //Recursive for triangular MPOs

    struct Sites
    {
        Sites();
        Sites(int leftSite,const iTEBDStateImp*);
        int leftSiteNumber;
        MPSSite* siteA;
        MPSSite* siteB;
        Bond*    bondA;
        Bond*    bondB;
        dVectorT* GammaA;  //Owned by MPSImp
        dVectorT* GammaB;  //Owned by MPSImp
        const DiagonalMatrixRT* lambdaA; //Owned by bonds
        const DiagonalMatrixRT* lambdaB; //Owned by bonds
    };

    const dVectorT& GammaA() const {return *s1.GammaA;}
    const dVectorT& GammaB() const {return *s1.GammaB;}
          dVectorT& GammaA()       {return *s1.GammaA;}
          dVectorT& GammaB()       {return *s1.GammaB;}
    const DiagonalMatrixRT& lambdaA() const {return *s1.lambdaA;}
    const DiagonalMatrixRT& lambdaB() const {return *s1.lambdaB;}

private:
    void InitSitesAndBonds(int D, double epsSV);

    mutable Sites s1; //ReCenter member function will build this structure.
};



inline MPSSite::dVectorT operator*(const MPSSite::dVectorT& gamma, const DiagonalMatrixRT& lambda)
{
    int d=gamma.size();
    MPSSite::dVectorT gl(d);
    for (int n=0; n<d; n++)
        gl[n]=gamma[n]*lambda;
    return gl;
}
inline MPSSite::dVectorT operator*(const DiagonalMatrixRT& lambda, const MPSSite::dVectorT& gamma)
{
    int d=gamma.size();
    MPSSite::dVectorT lg(d);
    for (int n=0; n<d; n++)
        lg[n]=lambda*gamma[n];
    return lg;
}

} //namespace
#endif // ITEBDSTATEIMP_H
