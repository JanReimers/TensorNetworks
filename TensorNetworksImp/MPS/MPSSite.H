#ifndef MATRIXPRODUCTSITE_H
#define MATRIXPRODUCTSITE_H

#include "TensorNetworksImp/Namespace.H"
#include "TensorNetworksImp/MPS/Bond.H"
#include "Containers/Vector3.H"
#include "oml/matrix.h"
#include <vector>

class MPOTests;
class MPSTests;
template <class T> class EigenSolver;
namespace TensorNetworks
{

class MPSSite : public virtual Site
{
    public:
//--------------------------------------------------------------------------------------
//
//  Construction and initialization zone
//
        MPSSite(Position lbr,Bond* leftBond, Bond* rightBond, int d, int D1, int D2);
        virtual ~MPSSite();
        void InitializeWith(State,int sgn);
        void Freeze(double s);
        void CloneState(const MPSSite* psi2);
        void NewBondDimensions(int D1, int D2, bool saveData=false);

//--------------------------------------------------------------------------------------
//
//  Access to information
//
        double GetSiteEnergy () const {return itsEmin;}
        double GetEGap       () const {return itsGapE;}
        double GetIterDE     () const {return itsIterDE;}
        bool   IsFrozen      () const {return isFrozen;}
        int    GetD1         () const {return itsD1;}
        int    GetD2         () const {return itsD2;}
//---------------------------------------------------------------------------------
//
//  Required for ground state calculations
//
        void     Refine (const MatrixCT& Heff,const Epsilons& eps);
        Matrix6CT GetHeff(const SiteOperator* so,const Vector3CT& L,const Vector3CT& R) const;
        const    Vector3CT& GetHeffCache(Direction lr) const
        {
            return lr==DLeft ? itsHLeft_Cache : itsHRightCache;
        }
        const    MatrixCT& GetRLCache(Direction lr) const
        {
            return lr==DLeft ? itsLeft_Cache : itsRightCache;
        }
        void     UpdateCache(const SiteOperator* so,const Vector3CT& HLeft,const Vector3CT& HRight); //update <Psi|H|Psi> L/R cahces
        void     UpdateCache(const MPSSite* Psi2   ,const MatrixCT&   Left,const MatrixCT&   Right); //update <Psi|Psi> L/R cahces

        MatrixCT  IterateLeft_F(const MPSSite* Psi2   ,const MatrixCT&  Fam1,bool cache=false) const;
        MatrixCT  IterateRightF(const MPSSite* Psi2   ,const MatrixCT&  Fam1,bool cache=false) const;
        Vector3CT IterateLeft_F(const SiteOperator* so,const Vector3CT& Fam1,bool cache=false);
        Vector3CT IterateRightF(const SiteOperator* so,const Vector3CT& Fam1,bool cache=false);
        MatrixCT  IterateF     (Direction lr ,const MatrixCT& M) const;

//---------------------------------------------------------------------------------
//
// Normalization
//
        void NormalizeQR  (Direction lr);
        void SVDNormalize (Direction lr,SVCompressorC*);
        void Canonicalize1(Direction lr,SVCompressorC*);
        void Canonicalize2(Direction lr,SVCompressorC*);

//--------------------------------------------------------------------------------------
//
//  Vary this site to be as close as possible to Site by minimizing ||this-Psi2||^2
//
        virtual void Optimize(const MPSSite* psi, const MatrixCT& L, const MatrixCT& R); // Minimize ||this-Psi2||^2

//--------------------------------------------------------------------------------------
//  Operate on MPS with an MPO:  |Psi> -> O|Psi>.
//  In general this will increase the bond dimension from D to D*Dw
//
        void  ApplyInPlace(const SiteOperator* so);
        void  Apply       (const SiteOperator* so, MPSSite* psiPrime);

//--------------------------------------------------------------------------------------
//
//    Density matrix calculations
//
        MatrixCT  CalculateOneSiteDM();
        MatrixCT InitializeTwoSiteDM(int m, int n     );
        MatrixCT    IterateTwoSiteDM(      MatrixCT& C);
        MatrixCT  FinializeTwoSiteDM(const MatrixCT& C);

//
//  Statusreporting
//
        bool IsNormalized(Direction lr,double eps) const;
        bool IsCanonical (Direction lr,double eps) const;
        char GetNormStatus(double eps) const;
        void Report    (std::ostream&) const;

        typedef std::vector<MatrixCT>     dVectorT;

private:
    friend class ::MPOTests;
    friend class ::MPSTests;
    friend class iTEBDStateImp;
    friend class Bond;


    Position WhereAreWe() const {return itsPosition;}

    MatrixCT  GetNorm         (Direction lr) const;
    MatrixCT  GetCanonicalNorm(Direction lr) const;
    MatrixCT  ReshapeBeforeSVD(Direction lr);
    void      ReshapeAfter_SVD(Direction lr,const MatrixCT& UV);
    void      Rescale(double norm);
    double    GetFrobeniusNorm() const;
    Matrix4CT GetTransferMatrix(Direction lr) const;
    void      Update      (const VectorCT& newAs);
    void      iNormalize(Direction lr); //Use transfer matrix eigenvalue to rescale.
    bool      HandleEdges(Direction lr);

    //
    //  lr is the normalization direction.  For normalization we need to access the right bond.
    //
    Bond* GetBond(Direction lr)
    {
        return lr==DLeft ? itsRightBond : itsLeft_Bond;
    }
    const Bond* GetBond(Direction lr) const
    {
        return lr==DLeft ? itsRightBond : itsLeft_Bond;
    }

     Direction Invert(Direction lr) const
    {
        return lr==DLeft ?  DRight : DLeft;
    }

    typedef dVectorT::      iterator  dIterT;
    typedef dVectorT::const_iterator cdIterT;

//
//  NormalizationStatus
//
    enum class NormStatus {M,A,B,GammaLeft,GammaRight}; //UnNormalized, Left Normalized, Right Normalized, Vidal Gamma-lambda-Gamma
//-----------------------------------------------------------------------------------
//
//  Internal contraction function
//
    void SVDTransfer(Direction lr,const DiagonalMatrixRT& s, const MatrixCT& UV);
    void SVDTransfer(Direction lr,const MatrixCT& UV);
    void TransferQR (Direction lr,const MatrixCT& R);

    dcmplx ContractAWFA (int w2,int i2, int j2,         const SiteOperator* so, const Vector3CT& Fam1) const;
    dcmplx ContractWFA  (int m, int w2, int i1, int j2, const SiteOperator* so, const Vector3CT& Fam1) const;
    dcmplx ContractFA   (int n, int w1, int i1, int j2,                         const Vector3CT& Fam1) const;
    dcmplx ContractBWFB (int w1,int i1, int j1,         const SiteOperator* so, const Vector3CT& Fap1) const;
    dcmplx ContractWFB  (int m, int w1, int i2, int j1, const SiteOperator* so, const Vector3CT& Fap1) const;
    dcmplx ContractFB   (int n, int w2, int i2, int j1,                         const Vector3CT& Fap1) const;

    dcmplx ContractWR   (int w1,int i2, int j2,const MatrixRT& W, int Dw2, const Vector3CT& R) const;

    MatrixCT ContractCA(int n2, const MatrixCT& C) const;
    void     Contract(dVectorT& newAs,const SiteOperator* so);
    MatrixCT ContractLRM(const MatrixCT& M, const MatrixCT& L, const MatrixCT& R) const;
   // MatrixCT Contract_RM(const MatrixCT& R, const MatrixCT& M) const;


    Bond* itsLeft_Bond;
    Bond* itsRightBond;
    int itsd; //2*S+1
    int itsD1,itsD2;  //Bond dimensions
    dVectorT itsMs;   //We need a separate matrix for each of the p basis states on one site.
    Vector3CT itsHLeft_Cache,itsHRightCache; //L/R tensors for <Psi|H|Psi>
    mutable MatrixCT itsLeft_Cache ,itsRightCache;  //L/R matrices for <Psi|Psi>
    EigenSolver<dcmplx>* itsEigenSolver;

    NormStatus itsNormStatus; //Indicates what the last norm command did. THis may or may not agree with the current data
    int        itsNumUpdates;
    bool       isFrozen; //Don't refine this site.
    double     itsEmin;
    double     itsGapE;  //Gap to the next excited state.
    double     itsIterDE;
    Position   itsPosition; //Left Bulk or Right
};

}

#endif // MATRIXPRODUCTSITE_H
