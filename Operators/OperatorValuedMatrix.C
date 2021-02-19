#include "OperatorValuedMatrix.H"
#include "NumericalMethods/LapackQRSolver.H"
#include "NumericalMethods/LapackSVDSolver.H"
#include "TensorNetworks/SVCompressor.H"
#include "oml/diagonalmatrix.h"

using std::cout;
using std::endl;

namespace TensorNetworks
{

template <class T> MatrixO<T>::MatrixO()
    : Matrix<OperatorElement<T> >()
    , itsd(0)
    , itsTruncationError(0)
    , itsUL(Full)
{}

template <class T> MatrixO<T>::MatrixO(int d,TriType ul)
    : Matrix<OperatorElement<T> >()
    , itsd(d)
    , itsTruncationError(0)
    , itsUL(ul)
{}

template <class T> MatrixO<T>::MatrixO(int d, const MatLimits& lim)
    : Matrix<OperatorElement<T> >(lim)
    , itsd(d)
    , itsTruncationError(0)
    , itsUL(Full)
{}

template <class T> MatrixO<T>::MatrixO(const Base& m)
    : Matrix<OperatorElement<T> >(m)
    , itsd(m(m.GetLimits().Row.Low,m.GetLimits().Col.Low).GetNumRows())
    , itsTruncationError(0)
    , itsUL(Full)
{
    CheckUL();
}

template <class T> MatrixO<T>::MatrixO(const MatrixO& m)
    : Matrix<OperatorElement<T> >(m)
    , itsd(m.itsd)
    , itsTruncationError(0)
    , itsUL(m.itsUL)
{
//    CheckUL();
}

template <class T> MatrixO<T>::MatrixO(MatrixO&& m)
    : Matrix<OperatorElement<T> >(m)
    , itsd(m.itsd)
    , itsTruncationError(0)
    , itsUL(m.itsUL)
{
//    CheckUL();
}


template <class T> MatrixO<T>::MatrixO(int Dw1, int Dw2,double S,TriType ul)
    : Matrix<OperatorElement<T> >(0,Dw1-1,0,Dw2-1)
    , itsd(2*S+1)
    , itsTruncationError(0)
    , itsUL(ul)
{
    OperatorElement<T> Z=OperatorZ(S);
    Fill(*this,Z);
}

template <class T> MatrixO<T>::MatrixO(int Dw1, int Dw2,int d,TriType ul)
    : Matrix<OperatorElement<T> >(0,Dw1-1,0,Dw2-1)
    , itsd(d)
    , itsTruncationError(0)
    , itsUL(ul)
{
    double S=0.5*(d-1);
    OperatorElement<T> Z=OperatorZ(S);
    Fill(*this,Z);
}


template <class T> MatrixO<T>::~MatrixO()
{
    //dtor
}

template <class T> typename MatrixO<T>::IIType MatrixO<T>::GetChi12() const
{
    return std::make_tuple(this->GetNumRows()-2,this->GetNumCols()-2);
}

template <class T> void MatrixO<T>::SetChi12(int X1,int X2,bool preserve_data)
{
    if (this->GetNumRows()!=X1+2 || this->GetNumCols()!=X2+2)
    {
        auto [X1o,X2o]=GetChi12(); //Save old size
        Base::SetLimits(0,X1+1,0,X2+1,preserve_data); //Save Data?

        double S=0.5*(itsd-1.0);
        OperatorElement<T> Z=OperatorZ(S);
        if (preserve_data)
        {
            // Zero new rows and columns (if any)
            for (int w1=X1o+2;w1<=X1+1;w1++)
            for (int w2=X2o+2;w2<=X2+1;w2++)
                (*this)(w1,w2)=Z;
        }
        else
        {
            Fill(*this,Z);
        }
    }
}

template <class T> void MatrixO<T>::CheckUL()
{
    double eps=1e-8;
    if (this->GetNumRows()<2 || this->GetNumCols()<2)
    {
        //itsUL=Lower; //Temporary kludge to get beyond the row/col matrix U/L ambiguity
    }
    else
    {
        if (IsLowerTriangular(*this,eps))
            itsUL=Lower;
        else if (IsUpperTriangular(*this,eps))
        {
            itsUL=Upper;
        }
        else
        {
//            cout << std::scientific << std::setprecision(1) << "Full matrix??=" << *this;
            itsUL=Full;
        }
    }
}

template <class T> void MatrixO<T>::Setd()
{
    MatLimits l=this->GetLimits();
    OperatorElement<T> e=(*this)(l.Row.Low,l.Col.Low);
    itsd=e.GetNumRows();
}
template <class T> void MatrixO<T>::SetUpperLower(TriType ul)
{
//    assert(this->GetNumRows()==1 || this->GetNumCols()==1);
    assert(ul==Upper || ul==Lower);
    itsUL=ul;
}

template <class T> MatrixO<T>& MatrixO<T>::operator=(const MatrixO<T>& m)
{
    Base::operator=(m);
    itsd=m.itsd;
    itsTruncationError=m.itsTruncationError;
    itsUL=m.itsUL;
    return *this;
}
template <class T> MatrixO<T>& MatrixO<T>::operator=(MatrixO<T>&& m)
{
    Base::operator=(m);
    itsd=m.itsd;
    itsTruncationError=m.itsTruncationError;
    itsUL=m.itsUL;
    return *this;
}

template <class T> MatrixO<T>& MatrixO<T>::operator*=(const T& s)
{
    for (index_t i:this->rows())
        for (index_t j:this->cols())
            (*this)(i,j)*=s;
    return *this;
}


template <class T> std::ostream& MatrixO<T>::PrettyPrint(std::ostream& os) const
{
    assert(itsd>1);
    os << std::fixed << std::setprecision(1) << this->GetLimits() << std::endl;
    for (index_t i:this->rows())
    {
        for (int m=0;m<itsd;m++)
        {
            for (index_t j:this->cols())
            {
                os << "[ ";
                for (int n=0;n<itsd;n++)
                    os << std::setw(4) <<(*this)(i,j)(m,n) << " ";
                os << "]  ";
            }
            os << std::endl;
        }
        os << std::endl;
    }
    return os;
}

template <class T> T MatrixO<T>::GetTrace(int a, int b, int c, int d) const
{
    T ret(0.0);
    for (int m=0;m<itsd;m++)
    for (int n=0;n<itsd;n++)
        ret+=conj((*this)(b,a))(m,n)*(*this)(c,d)(m,n);
    return ret/itsd;
}

template <class T> bool MatrixO<T>::IsOrthonormal(Direction lr,double eps) const
{
    return IsUnit(GetOrthoMatrix(lr),eps);
}

template <class T> Matrix<T> MatrixO<T>::GetOrthoMatrix(Direction lr) const
{
    Matrix<T> O;
    switch (lr)
    {
    case DLeft:
        O.SetLimits(this->GetColLimits(),this->GetColLimits());
        Fill(O,T(0.0));
        for (index_t b:this->cols())
        for (index_t c:this->cols())
            for (index_t a:this->rows())
                O(b,c)+=GetTrace(b,a,a,c);
        break;
    case DRight:
        O.SetLimits(this->GetRowLimits(),this->GetRowLimits());
        Fill(O,T(0.0));
        for (index_t b:this->rows())
        for (index_t c:this->rows())
            for (index_t a:this->cols())
                O(b,c)+=GetTrace(a,b,c,a);
        break;
    }
    return O;
}

//
//  THis routine is rather tricky because are actually 12 permutations here
//  Left/Right * Upper/Lower * Column/Rectangular/Row matrix.
//  For the column and row cases the V-block still needs to have one column or row.
//
template <class T> MatrixO<T> MatrixO<T>::GetV(Direction lr) const
{
    const MatLimits& l=this->GetLimits();
    int rl=l.Row.Low +1; //shifted first row.
    int rh=l.Row.High-1;
    int cl=l.Col.Low +1;
    int ch=l.Col.High-1;
    if (rl>l.Row.High) rl=l.Row.High; //enforce at least one row
    if (rh<l.Row.Low ) rh=l.Row.Low;  //enforce at least one row
    if (cl>l.Col.High) cl=l.Col.High; //enforce at least one column
    if (ch<l.Col.Low ) ch=l.Col.Low;  //enforce at least one column
    MatLimits lv;
    switch (lr)
    {
    case DLeft:
        switch(itsUL)
        {
        case Upper:
            lv=MatLimits(l.Row.Low,rh,l.Col.Low,ch);
            break;
        case Lower:
            lv=MatLimits(rl,l.Row.High,cl,l.Col.High);
            break;
        default:
            assert(false);
        }
        break;
    case DRight:
        switch(itsUL)
        {
        case Upper:
            lv=MatLimits(rl,l.Row.High,cl,l.Col.High);
            break;
        case Lower:
            lv=MatLimits(l.Row.Low,rh,l.Col.Low,ch);
            break;
        default:
            assert(false);
        }
        break;
    }
    assert(lv.GetNumRows()>0);
    assert(lv.GetNumCols()>0);
    return this->SubMatrix(lv);
}

template <class T> void MatrixO<T>::SetV(Direction lr,const MatrixO& V)
{
    //  If L has less columns than the Ws then we need to reshape the whole site.
    //  Typically this will happen at the edges of the lattice.
    //
    auto [X1,X2]=GetChi12();
    int nc=V.GetNumCols();
    int nr=V.GetNumRows();
    switch (lr)
    {
    case DLeft:
        if (nc-1<X2)
            SetChi12(X1,nc-1,true); //we must save the old since V only holds part of W
        break;
    case DRight:
        if (nr-1<X1)
        {
            VectorOR lastRow=this->GetRow(X1+1);
            assert(lastRow.size());
            SetChi12(nr-1,X2,true); //we must save the old since V only holds part of W
//            cout << "X2,lastRow=" << X2 << " " << lastRow.GetLimits() << endl;
            this->GetRow(nr)=lastRow.SubVector(0,X2+1);

        }
        break;
    }

    for (index_t i:V.rows())
        for (index_t j:V.cols())
            (*this)(i,j)=V(i,j);
}

template <class T> Matrix<T> MatrixO<T>::Flatten(Direction lr) const
{
    int rl=this->GetLimits().Row.Low;
    int cl=this->GetLimits().Col.Low;
    Matrix<T> F;
    switch (lr)
    {
    case DLeft:
    {
        int Dw1=itsd*itsd*this->GetNumRows();
        F.SetLimits(VecLimits(rl,rl+Dw1-1),this->GetLimits().Col);
        for (index_t j:this->cols())
        {
            int w=rl;
            for (index_t i:this->rows())
                for (int m=0; m<itsd; m++)
                    for (int n=0; n<itsd; n++)
                        F(w++,j)=(*this)(i,j)(m,n);
        }
    }
    break;
    case DRight:
    {
        int Dw2=itsd*itsd*this->GetNumCols();
        F.SetLimits(this->GetLimits().Row,VecLimits(cl,cl+Dw2-1));
        for (index_t i:this->rows())
        {
            int w=cl;
            for (index_t j:this->cols())
                for (int m=0; m<itsd; m++)
                    for (int n=0; n<itsd; n++)
                        F(i,w++)=(*this)(i,j)(m,n);
        }
    }
    break;
    }
    return F;
}


template <class T> void MatrixO<T>::UnFlatten(const Matrix<T>& F)
{
    int Nr=this->GetNumRows();
    int Nc=this->GetNumCols();
    MatLimits l=this->GetLimits();
    if (F.GetNumRows()==itsd*itsd*Nr)
    {
        if (F.GetNumCols()<Nc)
            this->SetLimits(l.Row,VecLimits(l.Col.Low,F.GetNumCols()+l.Col.Low-1),true);
        for (index_t j:this->cols())
        {
            int w=this->GetLimits().Row.Low;
            for (index_t i:this->rows())
                for (int m=0; m<itsd; m++)
                    for (int n=0; n<itsd; n++)
                        (*this)(i,j)(m,n)=F(w++,j);
        }

    }
    else if (F.GetNumCols()==itsd*itsd*this->GetNumCols())
    {
        if (F.GetNumRows()<Nr)
            this->SetLimits(VecLimits(l.Row.Low,F.GetNumRows()+l.Row.Low-1),l.Col,true);

        for (index_t i:this->rows())
        {
            int w=this->GetLimits().Col.Low;
            for (index_t j:this->cols())
                for (int m=0; m<itsd; m++)
                    for (int n=0; n<itsd; n++)
                        (*this)(i,j)(m,n)=F(i,w++);
        }
    }
    else
        assert(false);

}

template <class T> typename MatrixO<T>::QXType MatrixO<T>::BlockQX(Direction lr) const
{
    LapackQRSolver <double>  solver;
    MatrixO   V=GetV(lr);
    MatLimits Vlim=V.ReBase(1,1);
    Matrix<T> Vf=V.Flatten(lr);
    assert(FrobeniusNorm(Vf)>0.0); //Make sure we didn't get all zeros
    Matrix<T> RL;
    double scale=1.0;
    int Dw1=V.GetNumRows();
    int Dw2=V.GetNumCols();
    MatLimits RLlim;
    VecLimits rlim=this->GetLimits().Row;
    VecLimits clim=this->GetLimits().Col;
    int RLlow;

    switch (itsUL)
    {
    case Upper:
        switch (lr)
        {
        case DLeft:
        {
            auto [Q,R]=solver.SolveThinQR(Vf);
            assert(IsUpperTriangular(R));
            V.UnFlatten(Q);
            RL=R;
            scale=RL(1,1);
            RLlim=MatLimits(VecLimits(clim.Low,Q.GetColLimits().High),clim);
            RLlow=Vlim.Col.Low;
        }
        break;
        case DRight:
        {
            auto [R,Q]=solver.SolveThinRQ(Vf);
            assert(IsUpperTriangular(R));
            V.UnFlatten(Q);
            RL=R;
            scale=RL(Dw1,RL.GetNumCols());
            RLlim=MatLimits(rlim,VecLimits(rlim.Low,Q.GetRowLimits().High));
            RLlow=Vlim.Row.Low;
       }
        break;
        }
        break;
    case Lower:
        switch (lr)
        {
        case DLeft:
        {
            auto [Q,L]=solver.SolveThinQL(Vf);
            assert(IsLowerTriangular(L));
            V.UnFlatten(Q);
            RL=L;
            scale=RL(RL.GetNumRows(),Dw2);
            RLlim=MatLimits(VecLimits(clim.Low,Q.GetColLimits().High),clim);
            RLlow=Vlim.Col.Low;
        }
        break;
        case DRight:
        {
            auto [L,Q]=solver.SolveThinLQ(Vf);
            assert(IsLowerTriangular(L));
            V.UnFlatten(Q);
            RL=L;
            scale=RL(1,1);
            RLlim=MatLimits(rlim,VecLimits(rlim.Low,Q.GetRowLimits().High));
            RLlow=Vlim.Row.Low;
        }
        break;
        }
        break;
    default:
        assert(false);
    }
    // Do some sanity checks before re-scaling.
    if (scale==0.0)
        scale=1.0;
    else
    {
        assert(fabs(scale)>0.0);
        assert(fabs(fabs(scale)-sqrt(itsd))<1e-15);
    }

    assert(!isnan(RL));
    assert(!isinf(RL));
    RL/=scale;
    V*=scale;
    V .ReBase(Vlim);
    RL.ReBase(RLlow,RLlow);
    Grow(RL,RLlim);
    return std::make_tuple(V,RL);
}

//
//  This is where the fixed limits (0....X+1) helps us.  We know exactly where M is.
//  M is from (1..X1)x(1..X2) inside RL regardless of the limits RL.
//  the M area of the RL matrix gets replaced by a unit matrix.
//
template <class T> Matrix<T> MatrixO<T>::ExtractM(Matrix<T>& RL) const
{
    int X1=RL.GetNumRows()-2;
    int X2=RL.GetNumCols()-2;
    if (X1<0)
        assert(X1>=0);
    if (X2<0)
        assert(X2>=0);
    Matrix<T> M(X1,X2); //One based
    for (int w1=1;w1<=X1;w1++)
    for (int w2=1;w2<=X2;w2++)
    {
        M(w1,w2)=RL(w1,w2);
        RL(w1,w2)= (w1==w2) ? 1 : 0;
    }
    return M;
}

//
//  add unit rows and columns to m until m has the limits: lim
//
void Grow(Matrix<double>& m,const MatLimits& lim)
{
    const VecLimits&  rl=lim.Row;
    const VecLimits&  cl=lim.Col;
    const VecLimits& mrl=m.GetLimits().Row;
    const VecLimits& mcl=m.GetLimits().Col;
    assert( rl.Low == cl.Low );
//    assert( rl.High== cl.High);
    assert(mrl.Low ==mcl.Low );
//    assert(mrl.High==mcl.High);
    assert( rl.Low <=mrl.Low );
    assert( rl.High>=mrl.High);
    assert( cl.High>=mcl.High);
    int delta_high=rl.High-mrl.High;
    assert(cl.High-mcl.High==delta_high);
    m.SetLimits(lim,true); //Save the data.
    //
    //  Everything lines up in the top left corner so the code is reasonably simple here
    //
    for (int i= rl.Low;i<mrl.Low;i++)
    {
        m(i,i)=1.0;
        for (int j=i+1;j<=cl.High;j++) m(i,j)=0.0;
        for (int j=i+1;j<=rl.High;j++) m(j,i)=0.0;
    }
    //
    //  Since m is not necessarily square this is non trivial
    //
    for (int di=1;di<=delta_high;di++)
    {
        int i1=mrl.High+di;
        int i2=mcl.High+di;
        for (int j=cl.Low;j<=i2;j++) m(i1,j)= 0.0; // Fill out bottom row
        for (int j=rl.Low;j<=i1;j++) m(j,i2)= 0.0; // FIll out right column
        m(i1,i2)=1.0;
    }

}

template <class T> typename MatrixO<T>::QXType MatrixO<T>::BlockSVD(Direction lr,const SVCompressorR* comp)
{
    //
    //  Block respecting QR/QL/RQ/LQ
    //
    auto [Q,RL]=BlockQX(lr);
    assert(IsUnit(Q.GetOrthoMatrix(lr),1e-14));
    //
    //  Isolate the M matrix and SVD/compress it.
    //
    Matrix<T> M=ExtractM(RL);
    LapackSVDSolver <double>  solver;
    auto [U,s,VT]=solver.SolveAll(M,1e-14); //Solves M=U * s * VT
    itsTruncationError=comp->Compress(U,s,VT);
    int Xs=s.GetDiagonal().size();
//    cout << "s=" << s.GetDiagonal() << endl;
    //
    //  Post processing:
    //      1) Get RLtrans ready for transfer to the neighbouring site
    //      2) Integrate U (or VT) into Q
    MatrixRT RLtrans; //THis gets transferred to the neighbouring site;
    switch (lr)
    {
    case DLeft:
        {
            int base=Q.GetColLimits().Low; //base depends on the combination Left/Right*Upper/Lower
            assert(base==0 || base==1);
            Matrix<T> sV=s*VT;
            Grow(sV,MatLimits(VecLimits(0,Xs+1),RL.GetRowLimits()));
            RLtrans=sV*RL;
            Grow(U,MatLimits(Q.GetColLimits(),VecLimits(base,Xs+base)));
            Q=MatrixO(Q*U);
            assert(IsUnit(Q.GetOrthoMatrix(lr),1e-14));
        }
        break;
    case DRight:
        {
            int base=Q.GetRowLimits().Low; //base depends on the combination Left/Right*Upper/Lower
            assert(base==0 || base==1);
            Matrix<T> Us=U*s;
            Grow(Us,MatLimits(RL.GetColLimits(),VecLimits(0,Xs+1)));
            RLtrans=RL*Us;
            Grow(VT,MatLimits(VecLimits(base,Xs+base),Q.GetRowLimits()));
            Q=MatrixO(VT*Q);
            assert(IsUnit(Q.GetOrthoMatrix(lr),1e-14));
        }
        break;
    }
    return std::make_tuple(Q,RLtrans);
}

void SVDShuffle(TriType ul,Direction lr, Matrix<double>& U, DiagonalMatrix<double>& s,Matrix<double>& VT, double eps)
{
    std::vector<index_t> reindex;
    switch (lr)
    {
    case DLeft:
        reindex=FindRowReIndex(ul,VT,eps); // Pass in the scalar matrix
        break;
    case DRight:
        reindex=FindColReIndex(ul,U,eps); // Pass in the scalar matrix
        break;
    }
    VT.ReIndexRows   (reindex);
    s .ReIndexRows   (reindex);
    U .ReIndexColumns(reindex);
}


template class MatrixO <double>;

} //namespace

#include "oml/src/matrix.cpp"
#define Type TensorNetworks::OperatorElement<double>

template class Matrix<Type>;
