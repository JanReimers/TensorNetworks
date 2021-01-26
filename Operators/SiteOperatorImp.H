#ifndef SITEOPERATORIMP_H
#define SITEOPERATORIMP_H

#include "TensorNetworks/Namespace.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "TensorNetworksImp/Typedefs.H"
#include "oml/matrix.h"

class MPOTests;

template <class T> Matrix<T> TensorProduct(const Matrix<T>& a, const Matrix<T>& b)
{
    int nr=a.GetNumRows()*b.GetNumRows();
    int nc=a.GetNumCols()*b.GetNumCols();
    Matrix<T> r(nr,nc);
    //
    //  These loops need to run in the same order as SiteOperatorImp::Combine
    //
    int i=1;
    for (index_t ia:a.rows())
    for (index_t ib:b.rows())
    {
        int j=1;
        for (index_t ja:a.cols())
        for (index_t jb:b.cols())
        {
            r(i,j)=a(ia,ja)*b(ib,jb);
            j++;
        }
        i++;
    }
    return r;
}



namespace TensorNetworks
{
//
//  Derive from this in order to load up your site MPO/W matrices
//
class OperatorClient
{
protected:
    OperatorClient() {}
    virtual ~OperatorClient() {}
public:
    virtual MatrixRT GetW   (int m, int n) const=0;
    virtual Dw12     GetDw12() const=0;
    friend class ::MPOTests;
};

class SiteOperatorImp : public virtual SiteOperator
{
    protected:
        typedef Matrix<MatrixRT> TensorT;
        SiteOperatorImp(int d); //Construct with identity operator
        SiteOperatorImp(int d, double S, SpinOperator so); //Construct with identity operator
        SiteOperatorImp(int d, const OperatorClient* O);
        SiteOperatorImp(int d, Direction lr , const MatrixRT& U, const DiagonalMatrixRT& expEvs);
        SiteOperatorImp(int d, const TensorT& W); //Construct with W operator
    public:
        virtual ~SiteOperatorImp();
        void SetNeighbours(SiteOperator* left, SiteOperator* right);

        virtual int             Getd   () const {return itsd;}
        virtual const Dw12&     GetDw12() const=0;
        virtual const MatrixRT& GetW(int m, int n) const =0;
        virtual const MatrixRT& GetiW(int m, int n) const
        {
            return itsWs(m+1,n+1);
        }
        //
        //  Contract MPOs together
        //
        virtual void Combine(const SiteOperator* O2,double factor);
        //
        //  SVD compress and normalize to reduce Dw
        //
        virtual void CompressStd   (Direction,const SVCompressorR*);
        virtual void CompressParker(Direction,const SVCompressorR*);
        //
        //  As per Parker et. al.
        //
        virtual void CanonicalForm(Direction LR);

        virtual void Report(std::ostream&) const;
        virtual char GetNormStatus(double eps) const;
        virtual double GetFrobeniusNorm() const;
        virtual double GetiFrobeniusNorm() const;
        virtual void Update()=0;
        virtual void CheckDws() const;
        virtual void SetLimits(); //Fix up itsDw12 by inspecting zeros in Wmn
        static  void SetLimits(Dw12& Dw,TensorT& Ws); //Fix up itsDw12 by inspecting zeros in Wmn
    private:
        friend class MPOTesting;
        friend class iMPOImp;
        void SetiW(int m, int n, const MatrixRT& W);

        void SVDTransfer(Direction lr,const DiagonalMatrixRT& s,const MatrixRT& UV);
        void  QLTransfer(Direction lr,const MatrixRT& L);

        MatrixRT Reshape(Direction lr,int offset) const;
        void     Reshape(Direction lr,int offset,const MatrixRT& Q);
        void     NewBondDimensions(int D1, int D2, bool saveData=false);

        bool     isOrthonormal(Direction lr,const MatrixRT& Q) const;
        void     AccumulateTruncationError(double err);
    protected:

        int      itsd; //2*S+1
        Dw12     itsDw;
        double   itsTruncationError;
        bool     isShapeDirty; //Matrices got reshaped
        bool     isData_Dirty; //Matrices got new data

        TensorT  itsWs;
        SiteOperatorImp* itsLeft_Neighbour;
        SiteOperatorImp* itsRightNeighbour;
};

}
#endif // SITEOPERATORIMP_H
