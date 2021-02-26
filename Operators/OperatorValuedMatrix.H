#ifndef OPERATORVALUEDMATRIX_H
#define OPERATORVALUEDMATRIX_H

#include "Operators/OperatorElement.H"
#include "TensorNetworks/Namespace.H"

namespace TensorNetworks
{

template <class T> class MatrixO : public Matrix<OperatorElement<T> >
{
    typedef OperatorElement<T> EType;
    typedef std::tuple<MatrixO  ,Matrix<T> > QXType;
    typedef std::tuple<index_t  ,index_t   > IIType;
public:
    typedef Matrix<EType> Base;
//
//  Construction zone
//
    MatrixO();
    MatrixO(int d,TriType);
    MatrixO(int d,TriType, const MatLimits&);
    MatrixO(int Dw1, int Dw2,double S,TriType);
    MatrixO(int Dw1, int Dw2,int d,TriType);
    MatrixO(int d,TriType,const Base   & m);
    MatrixO(const MatrixO& m);
    virtual ~MatrixO();
//
//  Change size and nominal upper/lower
//
    void SetChi12(int X1,int X2,bool preserve_data);
    void SetUpperLower(TriType ul); //Row and column matrices have no way of knowing
//
// ASsign from overloaded operators
//
    template <class A, Store M, Data D> MatrixO(int d, TriType ul,const Indexable <EType,A,M,D,MatrixShape>& m)
    : Base(m)
    , itsd(d)
    , itsTruncationError(0)
    , itsUL(ul)
    {
    }
    template <class A, Store M, Data D> const MatrixO& operator=(const Indexable <EType,A,M,D,MatrixShape>& m)
    {
        Base::operator=(m);
        Setd();
        // TODO no way to transfer ul.
        return *this;
    }
//
//  Rescale
//
    MatrixO<T>& operator*=(const T& s);
//
//  Access to information
//
    IIType    GetChi12() const;
    int       Getd() const {return itsd;}
    double    GetS() const {return dtoS(itsd);}
    double    GetTruncationError() const {return itsTruncationError;}
    TriType   GetUpperLower() const {return itsUL;}
    T         GetTrace(int a, int b, int c, int d) const; //<Wab,Wcd>
    double    GetFrobeniusNorm() const;
    Matrix<T> GetOrthoMatrix(Direction) const;
    bool      IsOrthonormal (Direction,double eps) const;
//
//  V block operations
//
    MatrixO   GetV     (Direction) const;
    void      SetV     (Direction,const MatrixO& V);
    QXType    BlockQX  (Direction) const;
    QXType    BlockSVD (Direction,const SVCompressorR* comp) ;
//
//  Flatten for QR and SVD decompositions
//
    Matrix<T> Flatten  (Direction) const;
    void      UnFlatten(const Matrix<T>& F); //Limits of V decide R/L*U/L



    friend std::ostream& operator<<(std::ostream& os, const MatrixO& m)
    {
        return m.PrettyPrint(os);
    }
    std::ostream& PrettyPrint(std::ostream& os) const;
private:
    void Setd();
//    void CheckUL();
    Matrix<T> ExtractM(Matrix<T>& RL) const;

    int     itsd;
    double  itsTruncationError; //Should live on the bond with the SVs.
    TriType itsUL; //*Nominal* Upper or lower triangular.
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

inline TriType Combine(TriType a, TriType b)
{
    TriType ret=a;
    if (a!=b)
    {
        if (a==Diagonal) ret=b;
        else if (b==Diagonal) ret=a;
        else if (a==Full || b==Full) ret=Full;
    }
    return ret;
}

template <class T> MatrixO<T> operator-(const MatrixO<T>& a, const MatrixO<T>& b)
{
    assert(a.GetLimits()==b.GetLimits());
    assert(a.Getd     ()==b.Getd     ());
    TriType ul=Combine(a.GetUpperLower(),b.GetUpperLower());
    MatrixO<T> ret(a.Getd(),ul,a.GetLimits());
    OperatorElement<T> Z=OperatorZ(a.Getd());
    Fill(ret,Z);
    for (index_t i:a.rows())
        for (index_t j:a.cols())
            ret(i,j)=a(i,j)-b(i,j);
    return ret;
}

template <class T> MatrixO<T> operator*(const MatrixO<T>& a, const MatrixO<T>& b)
{
    assert(a.GetColLimits()==b.GetRowLimits());
    assert(a.Getd        ()==b.Getd        ());
    TriType ul=Combine(a.GetUpperLower(),b.GetUpperLower());
    const typename MatrixO<T>::Base& ma(a),mb(b);
    MatrixO<T> ret(a.Getd(),ul,ma*mb);
    return ret;
}

template <class T> MatrixO<T> operator*(const Matrix<T>& a, const MatrixO<T>& b)
{
    assert(a.GetColLimits()==b.GetRowLimits());
    const typename MatrixO<T>::Base mb(b);
    MatrixO<T> ret(b.Getd(),b.GetUpperLower(),a*mb);
    return ret;
}

template <class T> MatrixO<T> operator*(const MatrixO<T>& a, const Matrix<T>& b)
{
    assert(a.GetColLimits()==b.GetRowLimits());
    const typename MatrixO<T>::Base& ma(a);
    MatrixO<T> ret(a.Getd(),a.GetUpperLower(),ma*b);
    return ret;
}
inline index_t Mirror(index_t i,const VecLimits& lim)
{
    assert(i>=lim.Low);
    assert(i<=lim.High);
    int dl=i-lim.Low;
    int dh=lim.High-i;
    int im=0;
    if (dl<=dh)
        im=lim.High-dl;
    else
        im=lim.Low+dh;
    return im;
}
template <class T> void Mirror(Matrix<T>& a)
{
    const MatLimits& lim=a.GetLimits();
    for (index_t i:a.rows())
        for (index_t j:a.cols(i))
        {
            index_t im=Mirror(i,lim.Row);
            index_t jm=Mirror(j,lim.Col);
            T temp=a(i,j);
            a(i,j)=a(im,jm);
            a(im,jm)=temp;
        }
}


void Grow(Matrix<double>& m,const MatLimits& lim);

inline VecLimits operator+(const VecLimits& a, const VecLimits& b)
{
    return VecLimits(a.Low,a.Low+a.size()+b.size()-1);
}

inline VecLimits iSum(const VecLimits& a, const VecLimits& b)
{
    return VecLimits(a.Low,a.Low+a.size()+b.size()-1-2);
}

inline MatLimits operator+(const MatLimits& a, const MatLimits& b)
{
   return  MatLimits(a.Row+b.Row,a.Col+b.Col);
}

inline MatLimits iSum(const MatLimits& a, const MatLimits& b)
{
   return  MatLimits(iSum(a.Row,b.Row),iSum(a.Col,b.Col));
}


inline bool IsInitialized(const Matrix<double>& a)
{
    bool ret=true;
    for (index_t i:a.rows())
        for (index_t j:a.cols())
            if (a(i,j)!=0.0 && log10(a(i,j))<-300)
            {
                ret=false;
                std::cout << "Unintialized matrix element a(" << i << "," << j << ")=" << a(i,j) << std::endl;
            }
    return ret;
}


template <class T> Matrix<T> TensorSum(const Matrix<T>& a, const Matrix<T>& b)
{
    const MatLimits& al=a.GetLimits();
    const MatLimits& bl=b.GetLimits();
    MatLimits rl=al+bl;
    Matrix<T> r(rl);
    Fill (r,T(0.0));

    //
    //  Transfer a block
    //
    int i=rl.Row.Low;
    int j=0;
    for (index_t ia:a.rows())
    {
        j=rl.Col.Low;
        for (index_t ja:a.cols())
            r(i,j++)=a(ia,ja);
        i++;
    }

    //
    //  Transfer b block
    //
    int jstart=j;
    for (index_t ib:b.rows())
    {
        int j=jstart;
        for (index_t jb:b.cols())
            r(i,j++)=b(ib,jb);
        i++;
    }
    return r;
}

template <class T> Matrix<T> TensorSumLeft(const Matrix<T>& a, const Matrix<T>& b)
{
    const MatLimits& al=a.GetLimits();
    const MatLimits& bl=b.GetLimits();
    assert(al.GetNumRows()==1);
    assert(bl.GetNumRows()==1);
    MatLimits rl(al.Row,al.Col+bl.Col);
    Matrix<T> r(rl);
    Fill (r,T(0.0));

    //
    //  Transfer a block
    //
    int i=rl.Row.Low;
    int j=rl.Col.Low;
    for (index_t ja:a.cols())
        r(i,j++)=a(i,ja);
    for (index_t jb:b.cols())
        r(i,j++)=b(i,jb);

    return r;
}

template <class T> Matrix<T> TensorSumRight(const Matrix<T>& a, const Matrix<T>& b)
{
    const MatLimits& al=a.GetLimits();
    const MatLimits& bl=b.GetLimits();
    assert(al.GetNumCols()==1);
    assert(bl.GetNumCols()==1);
    MatLimits rl(al.Row+bl.Row,al.Col);
    Matrix<T> r(rl);
    Fill (r,T(0.0));

    //
    //  Transfer a block
    //
    int i=rl.Row.Low;
    int j=rl.Col.Low;
    for (index_t ia:a.rows())
        r(i++,j)=a(ia,j);
    for (index_t ib:b.rows())
        r(i++,j)=b(ib,j);

    return r;
}

template <class T> MatrixO<T> iTensorSum(const MatrixO<T>& a, const MatrixO<T>& b)
{
    int d=a.Getd();
    assert(d==b.Getd());
    TriType ul=Combine(a.GetUpperLower(),b.GetUpperLower());
    const MatLimits& al=a.GetLimits();
    const MatLimits& bl=b.GetLimits();
    MatLimits rl=iSum(al,bl);
    MatrixO<T> r(d,ul,rl);
    Fill (r,0.0);

    //
    //  Top and bottom rows
    //
    int i1=rl.Row.Low;
    int in=rl.Row.High;
    int j=rl.Col.Low;
    r(i1,j  )=OperatorElement<T>(d,1.0);
    r(in,j++)=a(al.Row.High,al.Col.Low)+b(bl.Row.High,bl.Col.Low);
    for (index_t ja:a.cols())
        if (ja>al.Col.Low && ja<al.Col.High)
        {
            r(i1,j  )=a(al.Row.Low ,ja);
            r(in,j++)=a(al.Row.High,ja);
        }
    for (index_t jb:b.cols())
        if (jb>bl.Col.Low && jb<bl.Col.High)
        {
            r(i1,j  )=b(bl.Row.Low ,jb);
            r(in,j++)=b(bl.Row.High,jb);
        }
    r(i1,j)=a(al.Row.Low,al.Col.High)+b(bl.Row.Low,bl.Col.High);
    r(in,j)=OperatorElement<T>(d,1.0);
    //
    //  Left and right columns
    //
    int j1=rl.Col.Low;
    int jn=rl.Col.High;
    int i=rl.Row.Low+1;
    for (index_t ia:a.rows())
        if (ia>al.Row.Low && ia<al.Row.High)
        {
            r(i  ,j1)=a(ia,al.Col.Low );
            r(i++,jn)=a(ia,al.Col.High);
        }
    for (index_t ib:b.rows())
        if (ib>bl.Row.Low && ib<bl.Row.High)
        {
            r(i  ,j1)=b(ib,bl.Col.Low );
            r(i++,jn)=b(ib,bl.Col.High);
        }
    //
    //  Now fill in the bulk
    //
    i=rl.Row.Low+1;
    for (index_t ia:a.rows())
        if (ia>al.Row.Low && ia<al.Row.High)
        {
            j=rl.Col.Low+1;
            for (index_t ja:a.cols())
                if (ja>al.Col.Low && ja<al.Col.High)
                    r(i,j++)=a(ia,ja);
            i++;
        }
    int jstart=j;
    for (index_t ib:b.rows())
        if (ib>bl.Row.Low && ib<bl.Row.High)
        {
            j=jstart;
            for (index_t jb:b.cols())
                if (jb>bl.Col.Low && jb<bl.Col.High)
                    r(i,j++)=b(ib,jb);
            i++;
        }

    return r;
}

template <class T> MatrixO<T> TensorSum(const MatrixO<T>& a, const MatrixO<T>& b)
{
    assert(a.Getd()==b.Getd());
    TriType ul=Combine(a.GetUpperLower(),b.GetUpperLower());
    const typename MatrixO<T>::Base& ma(a),mb(b);
    MatrixO<T> ret(a.Getd(),ul,TensorSum(ma,mb));
    return ret;
}


template <class T> MatrixO<T> TensorProduct(const MatrixO<T>& a, const MatrixO<T>& b)
{
    assert(a.Getd        ()==b.Getd        ());
    TriType ul=Combine(a.GetUpperLower(),b.GetUpperLower());
    const typename MatrixO<T>::Base& ma(a),mb(b);
    MatrixO<T> ret(a.Getd(),ul,TensorProduct(ma,mb));
    return ret;
}

//
//  This is a regular matrix product in the internal space (w1,w2) and a tensor product in
//  physical space (n,m).
//
template <class T> MatrixO<T> HorizontalProduct(const MatrixO<T>& a, const MatrixO<T>& b)
{
    assert(a.GetColLimits()==b.GetRowLimits());
    // Establish limits/d/ul
    MatLimits lim(a.GetRowLimits(),b.GetColLimits());
    int d=a.Getd()*b.Getd();
    TriType ul=Combine(a.GetUpperLower(),b.GetUpperLower());
    // Create empty results matrix
    MatrixO<T> ret(d,ul,lim);
    OperatorElement<T> Z=OperatorZ(d);
    Fill(ret,Z);
    //  Do the regular matrix product with tensor products for each element product.
    for (index_t w1:a.rows())
        for (index_t w3:b.cols())
            for (index_t w2:a.cols())
                ret(w1,w3)+=TensorProduct(a(w1,w2),b(w2,w3));
    return ret;
}



template <class T> void Fill(MatrixO<T>& m, T f)
{
    OperatorElement<T> foe(m.Getd(),f);
    Fill(m,foe);
}

template <class T> void Fill(MatrixO<T>& m, const OperatorElement<T>& f)
{
    assert(m.Getd()==f.Getd());
    for (index_t i:m.rows())
        for (index_t j:m.cols())
            m(i,j)= f;
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
