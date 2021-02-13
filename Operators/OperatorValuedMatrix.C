#include "OperatorValuedMatrix.H"
#include "NumericalMethods/LapackQRSolver.H"

namespace TensorNetworks
{

template <class T> MatrixO<T>::MatrixO()
    : Matrix<OperatorElement<T> >()
    , itsd(0)
    , itsUL(Full)
{}

template <class T> MatrixO<T>::MatrixO(const Base& m)
    : Matrix<OperatorElement<T> >(m)
    , itsd(m(m.GetLimits().Row.Low,m.GetLimits().Col.Low).GetNumRows())
    , itsUL(Full)
{
    CheckUL();
}

template <class T> MatrixO<T>::MatrixO(const MatrixO& m)
    : Matrix<OperatorElement<T> >(m)
    , itsd(m.itsd)
    , itsUL(Full)
{
    CheckUL();
}

template <class T> MatrixO<T>::MatrixO(MatrixO&& m)
    : Matrix<OperatorElement<T> >(m)
    , itsd(m.itsd)
    , itsUL(Full)
{
    CheckUL();
}


template <class T> MatrixO<T>::MatrixO(int Dw1, int Dw2,double S)
    : Matrix<OperatorElement<T> >(0,Dw1-1,0,Dw2-1)
    , itsd(2*S+1)
    , itsUL(Full)
{
    OperatorElement<T> Z=OperatorZ(S);
    Fill(*this,Z);
}


template <class T> MatrixO<T>::~MatrixO()
{
    //dtor
}
template <class T> void MatrixO<T>::CheckUL()
{
    if (IsLowerTriangular(*this))
        itsUL=Lower;
    else if (IsUpperTriangular(*this))
        itsUL=Upper;
    else
        itsUL=Full;

    MatLimits l=this->GetLimits();
    OperatorElement<T> e=(*this)(l.Row.Low,l.Col.Low);
    itsd=e.GetNumRows();
}
template <class T> MatrixO<T>& MatrixO<T>::operator=(const MatrixO<T>& m)
{
    Base::operator=(m);
    CheckUL();
    return *this;
}
template <class T> MatrixO<T>& MatrixO<T>::operator=(MatrixO<T>&& m)
{
    Base::operator=(m);
    CheckUL();
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
    os << std::fixed << std::setprecision(1) << std::endl;
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


template <class T> MatrixO<T> MatrixO<T>::GetV(Direction lr) const
{
    MatrixO<T> V;
    const MatLimits& l=this->GetLimits();
    MatLimits lv;
    switch (lr)
    {
    case DLeft:
        switch(itsUL)
        {
        case Upper:
            lv=MatLimits(l.Row.Low,l.Row.High-1,l.Col.Low,l.Col.High-1);
            break;
        case Lower:
            lv=MatLimits(l.Row.Low+1,l.Row.High,l.Col.Low+1,l.Col.High);
            break;
        default:
            assert(false);
        }
        break;
    case DRight:
        switch(itsUL)
        {
        case Upper:
            lv=MatLimits(l.Row.Low+1,l.Row.High,l.Col.Low+1,l.Col.High);
            break;
        case Lower:
            lv=MatLimits(l.Row.Low,l.Row.High-1,l.Col.Low,l.Col.High-1);
            break;
        default:
            assert(false);
        }
        break;
    }
    return this->SubMatrix(lv);
}

template <class T> void MatrixO<T>::SetV(const MatrixO& V)
{
    for (index_t i:V.rows())
        for (index_t j:V.cols())
            (*this)(i,j)=V(i,j);
    CheckUL();
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
    if (F.GetNumRows()==itsd*itsd*this->GetNumRows())
    {
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
    Matrix<T> RL;
    double scale=1.0;
    int Dw1=V.GetNumRows();
    int Dw2=V.GetNumCols();
    switch (itsUL)
    {
    case Upper:
        switch (lr)
        {
        case DLeft:
        {
            auto [Q,R1]=solver.SolveThinQR(Vf);
            V.UnFlatten(Q);
            RL=R1;
            scale=RL(1,1);
        }
        break;
        case DRight:
        {
            auto [R,Q]=solver.SolveThinRQ(Vf);
            V.UnFlatten(Q);
            RL=R;
            scale=RL(Dw1,Dw2);
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
            V.UnFlatten(Q);
            RL=L;
            scale=RL(Dw1,Dw2);
        }
        break;
        case DRight:
        {
            auto [L,Q]=solver.SolveThinLQ(Vf);
            V.UnFlatten(Q);
            RL=L;
            scale=RL(1,1);
        }
        break;
        }
        break;
    default:
        assert(false);
    }
    assert(fabs(scale)-itsd<1e-15);
    RL/=scale;
    V*=scale;
    V .ReBase(Vlim);
    RL.ReBase(Vlim);
    return std::make_tuple(V,RL);
}


template class MatrixO <double>;

} //namespace

#include "oml/src/matrix.cpp"
#define Type TensorNetworks::OperatorElement<double>

template class Matrix<Type>;
