#ifndef ITEBDSTATEIMP_H
#define ITEBDSTATEIMP_H

#include "TensorNetworks/iTEBDState.H"
#include "TensorNetworks/Typedefs.H"
#include "TensorNetworksImp/MPSImp.H"

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
        iTEBDStateImp(int L,double S, int D,double normEps,double epsSV);
        virtual ~iTEBDStateImp();
        virtual void InitializeWith(State);
        virtual void ReCenter(int isite);
 //--------------------------------------------------------------------------------------
//
//  Access to information
//
        virtual int GetL() const {return MPSImp::GetL();}
        virtual int Getp() const {return MPSImp::Getp();}
        virtual int GetD() const {return MPSImp::GetMaxD();}
//--------------------------------------------------------------------------------------
//
//  Normalization members with and without SVD compression
//
        virtual void Canonicalize(Direction);  //Vidals L-G-L-G-L canonical form
        virtual void Normalize(Direction LR);
        virtual ONErrors Orthogonalize(SVCompressorC*);

        virtual double GetExpectation (const MPO* o) const;
        virtual double GetExpectationmmnn (const Matrix4RT& Hlocal) const;
        virtual double GetExpectationmnmn (const Matrix4RT& expH) const;
        virtual void Apply(const Matrix4RT& expH,SVCompressorC*);
        virtual void Report(std::ostream&) const;
        virtual std::string GetNormStatus () const {return MPSImp::GetNormStatus();};

    protected:
        using dVectorT=MPSSite::dVectorT;
        typedef std::tuple<dVectorT,DiagonalMatrixRT> GLType;
        typedef std::tuple<MatrixCT,MatrixCT> MMType;
        typedef std::tuple<MatrixCT,double  > MdType;

        int GetModSite(int isite) const;
//        const DiagonalMatrixRT& GetLambda(int isite) const;
//        const         MatrixCT& GetGamma (int isite,int n) const ;
        virtual ONErrors GetOrthonormalityErrors() const;
        Matrix4CT GetTransferMatrix(Direction lr) const;
        Matrix4CT GetTransferMatrix(const dVectorT& M) const;
        Matrix4CT GetTransferMatrix(const Matrix4CT& theta) const;
        MatrixCT  GetNormMatrix(Direction lr,const dVectorT& M) const; //Er*I or I*El

        ONErrors UnpackOrthonormal(const dVectorT& gammap,DiagonalMatrixRT& lambdap,SVCompressorC*);
        GLType Orthogonalize(dVectorT& gamma, const DiagonalMatrixRT& lambda);
        MdType GetEigenMatrix(TensorNetworks::Direction lr, const Matrix4CT& theta);
        MMType Factor(const MatrixCT m);

        struct Sites
        {
            Sites();
            Sites(int leftSite,iTEBDStateImp*);
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

        Sites s1; //ReCenter member function will build this structure.
};

}
#endif // ITEBDSTATEIMP_H
