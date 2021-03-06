#ifndef IMATRIXPRODUCTSITE_H
#define IMATRIXPRODUCTSITE_H

#include "TensorNetworksImp/Namespace.H"
#include "TensorNetworksImp/MPS/Bond.H"
#include "TensorNetworksImp/iMPS/Tensor3.H"
#include "Containers/Vector3.H"
#include "oml/matrix.h"
#include <vector>

template <class T> class EigenSolver;
namespace TensorNetworks
{

class iMPSSite : public virtual Site
{
public:
//--------------------------------------------------------------------------------------
//
//  Construction and initialization zone
//
    iMPSSite(Bond* leftBond, Bond* rightBond, int d, int D);
    virtual ~iMPSSite();
    void InitializeWith(State,int sgn);
    void NewBondDimensions(int D1, int D2, bool saveData=false);

//--------------------------------------------------------------------------------------
//
//  Access to information
//
    double GetSiteEnergy () const {return itsEmin;}
    double GetEGap       () const {return itsGapE;}
    double GetIterDE     () const {return itsIterDE;}
    int    GetD1         () const {return itsD1;}
    int    GetD2         () const {return itsD2;}
//---------------------------------------------------------------------------------
//
//  Required for ground state calculations
//
    double RefineOneSite (const SiteOperator*,                         const Epsilons& eps);
    double Refine        (const SiteOperator*,iMPSSite* left_neighbour,const Epsilons& eps);
    double GetExpectation(const SiteOperator*) const;
//---------------------------------------------------------------------------------
//
// Normalization
//
    virtual void NormalizeQR  (Direction lr);
    virtual void NormalizeSVD (Direction lr,SVCompressorC*);
//        void Canonicalize1(Direction lr,SVCompressorC*);
//        void Canonicalize2(Direction lr,SVCompressorC*);

    virtual void SVDTransfer(Direction lr,const DiagonalMatrixRT& s, const MatrixCT& UV);
    virtual void SVDTransfer(Direction lr,const MatrixCT& UV);
    virtual void TransferQR (Direction lr,const MatrixCT& R);

    virtual void   InitQRIter();
    virtual double QRStep(Direction lr,double eps);
    virtual void   SaveAB_CalcLR(Direction lr); //save {A,B}=M and {L,R}=Transfer matrix eigen vector.
//
//  Statusreporting
//
    bool IsNormalized(Direction lr,double eps) const;
    bool IsCanonical (Direction lr,double eps) const;
    char GetNormStatus(double eps) const;
    void Report    (std::ostream&) const;

    typedef std::vector<MatrixCT>     dVectorT;

private:
    MatrixCT  GetNorm         (Direction lr) const;
    MatrixCT  GetCanonicalNorm(Direction lr) const;
//    MatrixCT  ReshapeBeforeSVD(Direction lr);
//    void      ReshapeAfter_SVD(Direction lr,const MatrixCT& UV);
//    void      Rescale(double norm);
    double    GetFrobeniusNorm() const;
//    Matrix4CT GetTransferMatrix(Direction lr) const;
//    void      Update      (const VectorCT& newAs);
//    void      iNormalize(Direction lr); //Use transfer matrix eigenvalue to rescale.
//    bool      HandleEdges(Direction lr);

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

//-----------------------------------------------------------------------------------
//
//  Internal contraction function
//
//    void SVDTransfer(Direction lr,const DiagonalMatrixRT& s, const MatrixCT& UV);
//    void SVDTransfer(Direction lr,const MatrixCT& UV);
//    void TransferQR (Direction lr,const MatrixCT& R);
//
//    dcmplx ContractAWFA (int w2,int i2, int j2,         const SiteOperator* so, const Vector3CT& Fam1) const;
//    dcmplx ContractWFA  (int m, int w2, int i1, int j2, const SiteOperator* so, const Vector3CT& Fam1) const;
//    dcmplx ContractFA   (int n, int w1, int i1, int j2,                         const Vector3CT& Fam1) const;
//    dcmplx ContractBWFB (int w1,int i1, int j1,         const SiteOperator* so, const Vector3CT& Fap1) const;
//    dcmplx ContractWFB  (int m, int w1, int i2, int j1, const SiteOperator* so, const Vector3CT& Fap1) const;
//    dcmplx ContractFB   (int n, int w2, int i2, int j1,                         const Vector3CT& Fap1) const;
//
//    dcmplx ContractWR   (int w1,int i2, int j2,const MatrixRT& W, int Dw2, const Vector3CT& R) const;
//
//    MatrixCT ContractCA(int n2, const MatrixCT& C) const;
//    void     Contract(dVectorT& newAs,const SiteOperator* so);
//    MatrixCT ContractLRM(const MatrixCT& M, const MatrixCT& L, const MatrixCT& R) const;
   // MatrixCT Contract_RM(const MatrixCT& R, const MatrixCT& M) const;


    Bond* itsLeft_Bond;
    Bond* itsRightBond;
    int itsd; //2*S+1
    int itsD1,itsD2;  //Bond dimensions
//    dVectorT itsMs;   //We need a separate matrix for each of the p basis states on one site.
    Tensor3  itsM,itsA,itsB; //M[n], and left and right versions A[n],B[n]
    MatrixCT itsG; //Gauge transform from the most recent normalization sweep.
    EigenSolver<dcmplx>* itsEigenSolver;

    int        itsNumUpdates;
    double     itsEmin;
    double     itsGapE;  //Gap to the next excited state.
    double     itsIterDE;
};

}

#endif // IMATRIXPRODUCTSITE_H
