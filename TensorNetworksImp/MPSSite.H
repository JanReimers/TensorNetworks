#ifndef MATRIXPRODUCTSITE_H
#define MATRIXPRODUCTSITE_H

#include "TensorNetworks/Enums.H"
#include "TensorNetworksImp/Typedefs.H"
#include "NumericalMethods/PrimeEigenSolver.H"
#include <vector>

class SiteOperator;
class Bond;
class SVCompressor;

class MPSSite
{
    typedef TensorNetworks::Matrix6CT Matrix6T;
    typedef TensorNetworks::Matrix4CT Matrix4T;
    typedef TensorNetworks::Vector3CT Vector3T;
    typedef TensorNetworks::Vector4CT Vector4T;
    typedef TensorNetworks::MatrixCT  MatrixCT;
    typedef TensorNetworks::VectorCT  VectorCT;
    typedef TensorNetworks::MatrixRT  MatrixRT;
    typedef TensorNetworks::VectorRT  VectorRT;
    typedef TensorNetworks::DiagonalMatrixRT DiagonalMatrixRT;
    typedef TensorNetworks::eType     eType;
    public:
//--------------------------------------------------------------------------------------
//
//  Conctruction and initialization zone
//
        MPSSite(TensorNetworks::Position lbr,Bond* leftBond, Bond* rightBond, int d, int D1, int D2);
        virtual ~MPSSite();
        void InitializeWith(TensorNetworks::State,int sgn);
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
//---------------------------------------------------------------------------------
//
//  Required for ground state calculations
//
        void     Refine (const MatrixCT& Heff,const Epsilons& eps);
        Matrix6T GetHeff(const SiteOperator* so,const Vector3T& L,const Vector3T& R) const;
        const    Vector3T& GetHeffCache(TensorNetworks::Direction lr) const
        {
            return lr==TensorNetworks::DLeft ? itsHLeft_Cache : itsHRightCache;
        }
        const    MatrixCT& GetRLCache(TensorNetworks::Direction lr) const
        {
            return lr==TensorNetworks::DLeft ? itsLeft_Cache : itsRightCache;
        }
        void     UpdateCache(const SiteOperator* so       ,const Vector3T& HLeft,const Vector3T& HRight); //update <Psi|H|Psi> L/R cahces
        void     UpdateCache(const MPSSite* Psi2,const MatrixCT&  Left,const MatrixCT&  Right); //update <Psi|Psi> L/R cahces

        MatrixCT IterateLeft_F(const MPSSite* Psi2,const MatrixCT& Fam1,bool cache=false) const;
        MatrixCT IterateRightF(const MPSSite* Psi2,const MatrixCT& Fam1,bool cache=false) const;
        MatrixCT IterateF     (TensorNetworks::Direction lr ,const MatrixCT& M) const;
        Vector3T IterateLeft_F(const SiteOperator*      so  ,const Vector3T& Fam1,bool cache=false);
        Vector3T IterateRightF(const SiteOperator*      so  ,const Vector3T& Fam1,bool cache=false);

//---------------------------------------------------------------------------------
//
// Normalization
//
        void SVDNormalize(TensorNetworks::Direction lr);
        void SVDNormalize(TensorNetworks::Direction lr,SVCompressor*);
        bool IsNormalized(TensorNetworks::Direction lr,double eps) const;
        bool SetCanonicalBondDimensions(int D1,int D2);
        void Canonicalize(TensorNetworks::Direction lr);

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
        char GetNormStatus(double eps) const;
        void Report    (std::ostream&) const;

//-----------------------------------------------------------------------------------
//
//  This stuff is for unti testing only.
//
private:
    friend class MPOTesting;
    friend class Bond;

    TensorNetworks::Position WhereAreWe() const {return itsPosition;}

    MatrixCT GetNorm(TensorNetworks::Direction lr) const;
    MatrixCT ReshapeBeforeSVD(TensorNetworks::Direction lr);
    void     ReshapeAfter_SVD(TensorNetworks::Direction lr,const MatrixCT& UV);
    void     Rescale(double norm);
    void     Update      (const VectorCT& newAs);

    static bool     IsUnit(const MatrixCT& m,double eps);

    //
    //  lr is the normalization direction.  For normalization we need to access the right bond.
    //
    Bond* GetBond(TensorNetworks::Direction lr)
    {
        return lr==TensorNetworks::DLeft ? itsRightBond : itsLeft_Bond;
    }


    friend class MatrixProductTesting;
    friend class iTEBDStateImp;

    typedef std::vector<MatrixCT>     dVectorT;
    typedef dVectorT::      iterator  dIterT;
    typedef dVectorT::const_iterator cdIterT;

//-----------------------------------------------------------------------------------
//
//  Internal contraction function
//
    void SVDTransfer(TensorNetworks::Direction lr,const DiagonalMatrixRT& s, const MatrixCT& UV);
    void SVDTransfer(TensorNetworks::Direction lr,const MatrixCT& UV);

    eType ContractAWFA (int w2,int i2, int j2,         const SiteOperator* so, const Vector3T& Fam1) const;
    eType ContractWFA  (int m, int w2, int i1, int j2, const SiteOperator* so, const Vector3T& Fam1) const;
    eType ContractFA   (int n, int w1, int i1, int j2,                         const Vector3T& Fam1) const;
    eType ContractBWFB (int w1,int i1, int j1,         const SiteOperator* so, const Vector3T& Fap1) const;
    eType ContractWFB  (int m, int w1, int i2, int j1, const SiteOperator* so, const Vector3T& Fap1) const;
    eType ContractFB   (int n, int w2, int i2, int j1,                         const Vector3T& Fap1) const;

    eType ContractWR   (int w1,int i2, int j2,const MatrixRT& W, int Dw2, const Vector3T& R) const;

    MatrixCT ContractCA(int n2, const MatrixCT& C) const;
    void     Contract(dVectorT& newAs,const SiteOperator* so);
    MatrixCT ContractLRM(const MatrixCT& M, const MatrixCT& L, const MatrixCT& R) const;
    MatrixCT Contract_RM(const MatrixCT& R, const MatrixCT& M) const;


    Bond* itsLeft_Bond;
    Bond* itsRightBond;
    int itsd; //2*S+1
    int itsD1,itsD2;  //Bond dimensions
    dVectorT itsMs;   //We need a separate matrix for each of the p basis states on one site.
    Vector3T itsHLeft_Cache,itsHRightCache; //L/R tensors for <Psi|H|Psi>
    mutable MatrixCT itsLeft_Cache ,itsRightCache;  //L/R matrices for <Psi|Psi>
    PrimeEigenSolver<eType> itsEigenSolver;

    int    itsNumUpdates;
    bool   isFrozen; //Don't refine this site.
    double itsEmin;
    double itsGapE;  //Gap to the next excited state.
    double itsIterDE;
    TensorNetworks::Position itsPosition; //Left Bulk or Right
};

#endif // MATRIXPRODUCTSITE_H
