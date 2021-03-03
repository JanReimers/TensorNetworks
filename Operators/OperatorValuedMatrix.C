#include "Operators/OperatorValuedMatrix.H"
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
    , itsForm(FUnknown)
{}

template <class T> MatrixO<T>::MatrixO(int d,MPOForm f)
    : Matrix<OperatorElement<T> >()
    , itsd(d)
    , itsForm(f)
{
}
template <class T> MatrixO<T>::MatrixO(int d,MPOForm f, const MatLimits& lim)
    : Matrix<OperatorElement<T> >(lim)
    , itsd(d)
    , itsForm(f)
{
}

template <class T> MatrixO<T>::MatrixO(int d,MPOForm f,const Base& m)
    : Matrix<OperatorElement<T> >(m)
    , itsd(d)
    , itsForm(f)
{
    // Find and fix any un-initialized elements
    OperatorElement<T> Z=OperatorZ(d);
    for (index_t i:this->rows())
    for (index_t j:this->cols())
    {
        if ((*this)(i,j).size()==0)
            (*this)(i,j)=Z;
    }

}

template <class T> MatrixO<T>::MatrixO(const MatrixO& m)
    : Matrix<OperatorElement<T> >(m)
    , itsd(m.itsd)
    , itsForm(m.itsForm)
{
}

template <class T> MatrixO<T>::MatrixO(int Dw1, int Dw2,double S,MPOForm f)
    : Matrix<OperatorElement<T> >(0,Dw1-1,0,Dw2-1)
    , itsd(2*S+1)
    , itsForm(f)

{
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

        OperatorElement<T> Z=OperatorZ(itsd);
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

template <class T> void MatrixO<T>::Setd()
{
    MatLimits l=this->GetLimits();
    OperatorElement<T> e=(*this)(l.Row.Low,l.Col.Low);
    itsd=e.GetNumRows();
}

template <class T> TriType  MatrixO<T>::GetMeasuredShape(double eps) const
{
    TriType ret=Full;
    if (IsDiagonal(*this,eps))
        ret=Diagonal;
    else if (this->GetNumRows()<2)
        ret=Row;
    else if (this->GetNumCols()<2)
        ret=Column;
    else if (IsUpperTriangular(*this,eps))
        ret=Upper;
    else if (IsLowerTriangular(*this,eps))
        ret=Lower;
    else
        ret=Full;

    return ret;
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
    os << std::fixed << std::setprecision(2) << this->GetLimits() << std::endl;
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

template <class T> double MatrixO<T>::GetFrobeniusNorm() const
{
    double fn=0.0;
    for (index_t i:this->rows())
    for (index_t j:this->cols())
        fn+=FrobeniusNorm((*this)(i,j));
    return fn;
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
    if      ((lr==DLeft && itsForm==RegularUpper) || (lr==DRight && itsForm==RegularLower))
        lv=MatLimits(l.Row.Low,rh,l.Col.Low,ch);
    else if ((lr==DLeft && itsForm==RegularLower) || (lr==DRight && itsForm== RegularUpper))
        lv=MatLimits(rl,l.Row.High,cl,l.Col.High);
    else if (itsForm==expH)
        lv=this->GetLimits();
    else
        assert(false);
//    else if  ((lr==DLeft && itsUL==Diagonal) || (lr==DRight && itsUL==Diagonal))
//        lv=MatLimits(l.Row.Low,rh,l.Col.Low,ch);

    assert(lv.GetNumRows()>0);
    assert(lv.GetNumCols()>0);
    return  MatrixO<T>(Getd(),itsForm,this->SubMatrix(lv));
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
        {
            VectorOR lastCol=this->GetColumn(X2+1);
            assert(lastCol.size());
            SetChi12(X1,nc-1,true); //we must save the old since V only holds part of W
            this->GetColumn(nc)=lastCol.SubVector(0,X1+1);
        }
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


template <typename T> inline int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template <class T> Matrix<T> MatrixO<T>::QX(Direction lr)
{
    MatrixO<T> Q;
    Matrix<T>  R;
    switch (itsForm)
    {
    case expH:
        std::tie(Q,R)=Full_QX(lr);
        break;
    case RegularUpper:
    case RegularLower:
        std::tie(Q,R)=BlockQX(lr);
        SetV(lr,Q);
        break;
    default:
        assert(false);
    }
    return R;
}

template <class T> typename MatrixO<T>::SVDType MatrixO<T>::SVD (Direction lr,const SVCompressorR* comp)
{
    SVDType ret;
    switch (itsForm)
    {
    case expH:
        ret=this->Full_SVD(lr,comp);
        break;
    case RegularUpper:
    case RegularLower:
        ret=this->BlockSVD(lr,comp);
        break;
    default:
        assert(false);
    }
    return ret;
}


template <class T> typename MatrixO<T>::QXType MatrixO<T>::BlockQX(Direction lr)
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
    int RLlow=0;

    switch (itsForm)
    {
    case RegularUpper:
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
    case RegularLower:
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
        if (fabs(fabs(scale)-sqrt(itsd))>=1e-15)
        {
//            cout << std::scientific << "Scale=" << std::setprecision(8) << scale << ", scale-root(d)="  << fabs(fabs(scale)-sqrt(itsd)) << endl;
//            cout << std::scientific << std::setprecision(3) << "V=" << V << endl;
//            cout << std::scientific << std::setprecision(3) << "Vf=" << Vf << endl;
//            cout << std::scientific << std::setprecision(3) << "RL=" << RL << endl;
            scale=sgn(scale)*sqrt(itsd);
        }
//        assert(fabs(fabs(scale)-sqrt(itsd))<1e-15);
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

template <class T> typename MatrixO<T>::QXType MatrixO<T>::Full_QX(Direction lr)
{
    assert(itsForm==expH);
    LapackQRSolver <double>  solver;
    Matrix<T> Wf=this->Flatten(lr);
    MatLimits Wlim=Wf.ReBase(1,1);
    assert(FrobeniusNorm(Wf)>0.0); //Make sure we didn't get all zeros
    Matrix <T> R,Q;

    switch (lr)
    {
        case DLeft:
        {
            std::tie(Q,R)=solver.SolveThinQR(Wf);
            R.ReBase(Wlim.Col.Low,Wlim.Col.Low);
        }
        break;
        case DRight:
        {
            std::tie(R,Q)=solver.SolveThinRQ(Wf);
            R.ReBase(Wlim.Row.Low,Wlim.Row.Low);
       }
        break;
    }
    assert(!isnan(R));
    assert(!isinf(R));
    assert(IsUpperTriangular(R));
    Q.ReBase(Wlim);
    this->UnFlatten(Q);
    double scale=sqrt(itsd);
    R/=scale;
    (*this)*=scale;
    return std::make_tuple(*this,R);
}

//
//  This is where the fixed limits (0....X+1) helps us.  We know exactly where M is.
//  M is from (1..X1)x(1..X2) inside RL regardless of the limits RL.
//  the M area of the RL matrix gets replaced by a unit matrix. BUt ...
//  we need Mp*RL to the have the same dimensions and the original RL
//
template <class T> Matrix<T> MatrixO<T>::ExtractM(Matrix<T>& RL,bool buildRp) const
{
    assert(IsInitialized(RL));
    int X1=RL.GetNumRows()-2;
    int X2=RL.GetNumCols()-2;
    if (X1<0)
        assert(X1>=0);
    if (X2<0)
        assert(X2>=0);
    assert(X2>=X1); //If this fails we need to shrink RL
    Matrix<T> M(X1,X2); //One based

    for (int w1=1;w1<=X1;w1++)
    for (int w2=1;w2<=X2;w2++)
        M(w1,w2)=RL(w1,w2);


    if (buildRp)
    {
        for (int w1=1;w1<=X1;w1++)
        for (int w2=1;w2<=X2;w2++)
            RL(w1,w2)= (w1==w2) ? 1 : 0;
        if (X2>X1)
        { //Fill in the rest of the unit matrix or X2>X1
            RL.SetLimits(0,X2+1,0,X2+1,true); //save data
            // Move the last row
            RL.GetRow(X2+1)=RL.GetRow(X1+1);
            for (int w1=X1+1;w1<=X2;w1++)
                for (int w2=0;w2<=X2+1;w2++)
                    RL(w1,w2)= (w1==w2) ? 1 : 0;
       }

    }
    assert(IsInitialized(M));
    assert(IsInitialized(RL));
    return M;
}

//
//  add unit rows and columns to m until m has the limits: lim
//
void Grow(Matrix<double>& m,const MatLimits& lim)
{
    assert(IsInitialized(m));
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
    assert(IsInitialized(m));

}

MatrixRT SolveRp(Direction lr, const MatrixRT& R, const MatrixRT& U, const DiagonalMatrixRT& s, const MatrixRT& VT)
{
    assert(Max(s.GetDiagonal())>1e-14);
    DiagonalMatrixRT sinv=1.0/s;
    MatrixRT Minv=Transpose(VT)*sinv*Transpose(U);
    int X1=Minv.GetNumRows();
    int X2=Minv.GetNumCols();
    Grow(Minv,MatLimits(0,X1+1,0,X2+1));
    MatrixRT Rp;
    switch (lr)
    {
    case DLeft:
        Rp=Minv*R;
        break;
    case DRight:
        Rp=R*Minv;
        break;
    }
    return Rp;
}

template <class T> typename MatrixO<T>::SVDType MatrixO<T>::FullMSVD(Direction lr,const SVCompressorR* comp)
{
    //
    //  Block respecting QR/QL/RQ/LQ
    //
    auto [Q,R]=Full_QX(lr);
    assert(IsInitialized(R));
    assert(IsUnit(Q.GetOrthoMatrix(lr),1e-14));
    //
    //  Isolate the M matrix and SVD/compress it.
    //
    Matrix<T> M=ExtractM(R,false);
    assert(IsInitialized(M));
    assert(IsInitialized(R));
    if (M.size()==0)
    {
        *this=Q;
        return std::make_tuple(0,DiagonalMatrixRT(),R);
    }
    LapackSVDSolver <double>  solver;
    auto [U,s,VT]=solver.SolveAll(M,1e-14); //Solves M=U * s * VT
    double truncationError=comp->Compress(U,s,VT);
    int Xs=s.GetDiagonal().size();

//    cout << std::fixed << std::setprecision(4) << "s=" << s.GetDiagonal() << endl;
    //
    //  Post processing:
    //      1) Get RLtrans ready for transfer to the neighbouring site
    //      2) Integrate U (or VT) into Q
    if (Xs>0) R=SolveRp(lr,R,U,s,VT);
    MatrixRT RLtrans; //THis gets transferred to the neighbouring site;
    switch (lr)
    {
    case DLeft:
        {
            Matrix<T> sV=s*VT;
            Grow(sV,MatLimits(VecLimits(0,Xs+1),R.GetRowLimits()));
            RLtrans=sV*R;
            Grow(U,MatLimits(Q.GetColLimits(),VecLimits(0,Xs+1)));
            Q=Q*U;
        }
        break;
    case DRight:
        {
            Matrix<T> Us=U*s;
            Grow(Us,MatLimits(R.GetColLimits(),VecLimits(0,Xs+1)));
            RLtrans=R*Us;
            Grow(VT,MatLimits(VecLimits(0,Xs+1),Q.GetRowLimits()));
            Q=VT*Q;
        }
        break;
    }
    assert(IsUnit(Q.GetOrthoMatrix(lr),1e-14));
    *this=static_cast<const Base&>(Q);
    return std::make_tuple(truncationError,s,RLtrans);
}

template <class T> typename MatrixO<T>::SVDType MatrixO<T>::Full_SVD(Direction lr,const SVCompressorR* comp)
{
    //
    //  Block respecting QR/QL/RQ/LQ
    //
    auto [Q,R]=Full_QX(lr);
    assert(IsInitialized(R));
    assert(IsUnit(Q.GetOrthoMatrix(lr),1e-14));
    LapackSVDSolver <double>  solver;
    MatLimits lim=R.ReBase(1,1);
    auto [U,s,VT]=solver.SolveAll(R,1e-14); //Solves R=U * s * VT
    double truncationError=comp->Compress(U,s,VT);
    U .ReBase(lim);
    s .ReBase(lim);
    VT.ReBase(lim);
//    cout << std::fixed << std::setprecision(4) << "s=" << s.GetDiagonal() << endl;
    //
    //  Post processing:
    //      1) Get RLtrans ready for transfer to the neighbouring site
    //      2) Integrate U (or VT) into Q
    MatrixRT RLtrans; //THis gets transferred to the neighbouring site;
    switch (lr)
    {
    case DLeft:
        {
            RLtrans=s*VT;
            Q=Q*U;
        }
        break;
    case DRight:
        {
            RLtrans=U*s;
            Q=VT*Q;
        }
        break;
    }
    assert(IsUnit(Q.GetOrthoMatrix(lr),1e-14));
    *this=static_cast<const Base&>(Q);
    return std::make_tuple(truncationError,s,RLtrans);
}

template <class T> typename MatrixO<T>::SVDType MatrixO<T>::BlockSVD(Direction lr,const SVCompressorR* comp)
{
    //
    //  Block respecting QR/QL/RQ/LQ
    //
    auto [Q,RL]=BlockQX(lr);
    assert(IsInitialized(RL));
    assert(IsUnit(Q.GetOrthoMatrix(lr),1e-14));
    //
    //  Isolate the M matrix and SVD/compress it.
    //
    Matrix<T> M=ExtractM(RL);
    assert(IsInitialized(M));
    assert(IsInitialized(RL));
    if (M.size()==0) return std::make_tuple(0.0,DiagonalMatrixRT(),RL);
    LapackSVDSolver <double>  solver;
    auto [U,s,VT]=solver.SolveAll(M,1e-14); //Solves M=U * s * VT
    double truncationError=comp->Compress(U,s,VT);
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
            Q=Q*U;
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
            Q=VT*Q;
        }
        break;
    }
    assert(IsUnit(Q.GetOrthoMatrix(lr),1e-14));
    SetV(lr,Q);
    return std::make_tuple(truncationError,s,RLtrans);
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
