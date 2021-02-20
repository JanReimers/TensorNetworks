#ifndef OPERATORVALUEDMATRIX_H
#define OPERATORVALUEDMATRIX_H

#include "Operators/OperatorElement.H"
#include "TensorNetworks/Namespace.H"

namespace TensorNetworks
{

template <class T> class MatrixO : public Matrix<OperatorElement<T> >
{
    typedef OperatorElement<T> EType;
    typedef Matrix<EType> Base;
    typedef std::tuple<MatrixO  ,Matrix<T> > QXType;
    typedef std::tuple<index_t  ,index_t   > IIType;
public:
    MatrixO();
    MatrixO(int d,TriType);
    MatrixO(int d, const MatLimits&);
    MatrixO(int Dw1, int Dw2,double S,TriType);
    MatrixO(int Dw1, int Dw2,int d,TriType);
    MatrixO(const Base   & m);
    MatrixO(const MatrixO& m);
    MatrixO(MatrixO&& m);
    virtual ~MatrixO();

    void SetChi12(int X1,int X2,bool preserve_data);
    void SetUpperLower(TriType ul); //Row and column matrices have no way of knowing (I think?)
    IIType GetChi12() const;

    MatrixO& operator=(const MatrixO& m);
    MatrixO& operator=(MatrixO&& m);
    template <class A, Store M, Data D> MatrixO(const Indexable <EType,A,M,D,MatrixShape>& m)
    : Base(m)
    {
        Setd();
    }
    template <class A, Store M, Data D> const MatrixO& operator=(const Indexable <EType,A,M,D,MatrixShape>& m)
    {
        Base::operator=(m);
        Setd();
        return *this;
    }
    MatrixO<T>& operator*=(const T& s);

    int       Getd() const {return itsd;}
    double    GetS() const {return 0.5*(itsd-1);}
    double    GetTruncationError() const {return itsTruncationError;}
    TriType   GetUpperLower() const {return itsUL;}
    T         GetTrace(int a, int b, int c, int d) const;
    double    GetFrobeniusNorm() const;
    Matrix<T> GetOrthoMatrix(Direction) const;
    bool      IsOrthonormal (Direction,double eps) const;
    MatrixO   GetV     (Direction) const;
    Matrix<T> Flatten  (Direction) const;
    QXType    BlockQX  (Direction) const;
    QXType    BlockSVD (Direction,const SVCompressorR* comp) ;

    void      SetV     (Direction,const MatrixO& V);
    void      UnFlatten(const Matrix<T>& F); //Limits of V decide R/L*U/L


    friend std::ostream& operator<<(std::ostream& os, const MatrixO& m)
    {
        return m.PrettyPrint(os);
    }
    std::ostream& PrettyPrint(std::ostream& os) const;
private:
    void Setd();
    void CheckUL();
    Matrix<T> ExtractM(Matrix<T>& RL) const;

    int     itsd;
    double  itsTruncationError; //Should live on the bond with the SVs.
    TriType itsUL; //Upper or lower triangular.
};

//typedef MatrixO<double> MatrixOR;
typedef MatrixO<dcmplx> MatrixOC;
typedef Vector<OperatorElement<double>> VectorOR;

template <class T> double MaxDelta(const MatrixO<T>& a, const MatrixO<T>& b)
{
    assert(a.GetLimits()==b.GetLimits());
    Matrix<T> ret(a.GetLimits());
    for (index_t i:a.rows())
        for (index_t j:a.cols())
            ret(i,j)=Max(fabs(a(i,j)-b(i,j)));
    return Max(fabs(ret));
}
template <class T> MatrixO<T> operator-(const MatrixO<T>& a, const MatrixO<T>& b)
{
    assert(a.GetLimits()==b.GetLimits());
    assert(a.Getd     ()==b.Getd     ());
    MatrixO<T> ret(a.Getd(),a.GetLimits());
    double S=0.5*(a.Getd()-1.0);
    OperatorElement<T> Z=OperatorZ(S);
    Fill(ret,Z);
    for (index_t i:a.rows())
        for (index_t j:a.cols())
            ret(i,j)=a(i,j)-b(i,j);
    return ret;
}

void Grow(Matrix<double>& m,const MatLimits& lim);

inline VecLimits operator*(const VecLimits& a, const VecLimits& b)
{
    assert(a.Low==b.Low);
    int size=a.size()*b.size();
    return VecLimits(a.Low,a.Low+size-1);
}

inline MatLimits operator*(const MatLimits& a, const MatLimits& b)
{
    return MatLimits(a.Row*b.Row,a.Col*b.Col);
}

template <class T> Matrix<T> TensorProduct(const Matrix<T>& a, const Matrix<T>& b)
{
    Matrix<T> r(a.GetLimits()*b.GetLimits());
    //
    //  These loops need to run in the same order as SiteOperatorImp::Product
    //
    int i=r.GetRowLimits().Low;
    for (index_t ib:b.rows())
    for (index_t ia:a.rows())
    {
        int j=r.GetColLimits().Low;
        for (index_t jb:b.cols())
        for (index_t ja:a.cols())
        {
            r(i,j)=a(ia,ja)*b(ib,jb);
            j++;
        }
        i++;
    }
    return r;
}

//
//  This is a regular matrix product in the internal space (w1,w2) and a tensor product in
//  physical space (n,m).
//
template <class T> MatrixO<T> HorizontalProduct(const MatrixO<T>& a, const MatrixO<T>& b)
{
    assert(a.GetColLimits()==b.GetRowLimits());
    MatLimits lim(a.GetRowLimits(),b.GetColLimits());
    int d=a.Getd()*b.Getd();
    MatrixO<T> ret(d,lim);
    double S=0.5*(d-1);
    OperatorElement<T> Z=OperatorZ(S);
    Fill(ret,Z);
    for (index_t w1:a.rows())
        for (index_t w3:b.cols())
            for (index_t w2:a.cols())
                ret(w1,w3)+=TensorProduct(a(w1,w2),b(w2,w3));
    return ret;
}

template <class T> bool IsTriangular(TriType ul,const Matrix<T>& m,double eps)
{
    bool ret=true;
    if (m.GetNumCols()>1 && m.GetNumRows()>1)
    {
        switch (ul)
        {
        case Upper:
            ret=IsUpperTriangular(m,eps);
            break;
        case Lower:
            ret=IsLowerTriangular(m,eps);
            break;
        case Full:
            ret=false;
        }
    }
    return ret;
}

template <class T> void Unit(MatrixO<T>& m)
{
    for (index_t i:m.rows())
        for (index_t j:m.cols())
            m(i,j)= (i==j) ? 1.0 : 0.0;
}

void SVDShuffle(TriType ul,Direction lr, Matrix<double>& U, DiagonalMatrix<double>& s,Matrix<double>& VT, double eps);

} //namespace

#endif // OPERATORVALUEDMATRIX_H
