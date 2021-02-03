#include "Operators/SiteOperatorImp.H"

namespace TensorNetworks
{


//
//  For LR = {Left/Right} we need to reshape with only {bottom right/top left} portion of matrix which
//  has the intrinsic portion of W.
//  In simple terms we just leave out the {first/last} row and column
//
MatrixRT SiteOperatorImp::ReshapeW(Direction lr) const
{
    MatrixRT A;
    switch (lr)
    {
    case DLeft:
    { //Leave out **first** row and column of W
        A.SetLimits(itsd*itsd*(itsDw.Dw1),itsDw.Dw2);
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                const MatrixRT& W=GetW(m,n);
                for (int w1=1; w1<=itsDw.Dw1; w1++,w++)
                    for (int w2=1; w2<=itsDw.Dw2; w2++)
                        A(w,w2)=W(w1,w2);
            }
        break;
    }
    case DRight:
    { //Leave out **last** row and column of W
        A.SetLimits(itsDw.Dw1,itsd*itsd*(itsDw.Dw2));
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                const MatrixRT& W=GetW(m,n);
                for (int w2=1; w2<=itsDw.Dw2; w2++,w++)
                    for (int w1=1; w1<=itsDw.Dw1; w1++)
                        A(w1,w)=W(w1,w2);
            }
        break;
    }
    }
    return A;
}

//
//  For LR = {Left/Right} we need to reshape with only {bottom right/top left} portion of matrix which
//  has the intrinsic portion of W.
//  In simple terms we just leave out the {first/last} row and column
//
MatrixRT SiteOperatorImp::ReshapeV(Direction lr) const
{
    MatrixRT A;
    MatLimits Vlim=GetV(lr,0,0).GetLimits();
    switch (lr)
    {
    case DLeft:
    { //Leave out **first** row and column of W
        A.SetLimits(itsd*itsd*Vlim.GetNumRows(),Vlim.GetNumCols());
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT V=GetV(lr,m,n);
                for (int w1=1; w1<=V.GetNumRows(); w1++,w++)
                    for (int w2=1; w2<=V.GetNumCols(); w2++)
                        A(w,w2)=V(w1,w2);
            }
        break;
    }
    case DRight:
    { //Leave out **last** row and column of W
        A.SetLimits(Vlim.GetNumRows(),itsd*itsd*Vlim.GetNumCols());
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT V=GetV(lr,m,n);
                for (int w2=1; w2<=V.GetNumCols(); w2++,w++)
                    for (int w1=1; w1<=V.GetNumRows(); w1++)
                        A(w1,w)=V(w1,w2);
            }
        break;
    }
    }
    return A;
}

void  SiteOperatorImp::ReshapeV(Direction lr,const MatrixRT& Q)
{
    switch (lr)
    {
    case DLeft:
    {
        //  If L has less columns than the Ws then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (Q.GetNumCols()+1<itsDw.Dw2)
            NewBondDimensions(itsDw.Dw1,Q.GetNumCols()+1,true);//we must save the old since Q only holds part of W
        MatLimits Vlim=GetV(lr,0,0).GetLimits();
        //Leave out **first** row and column of W
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT V(Vlim); //TODO just need limits here
                for (int w1=1; w1<=V.GetNumRows(); w1++,w++)
                    for (int w2=1; w2<=V.GetNumCols(); w2++)
                        V(w1,w2)=Q(w,w2);
                SetV(lr,m,n,V);
            }
        break;
    }
    case DRight:
    {
        //  If Vdagger has less rows than the Ws then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (Q.GetNumRows()+1<itsDw.Dw1)
            NewBondDimensions(Q.GetNumRows()+1,itsDw.Dw2,true);//we must save the old since Q only holds part of W
        MatLimits Vlim=GetV(lr,0,0).GetLimits();
        //Leave out **last** row and column of W
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT V(Vlim);
                for (int w2=1; w2<=V.GetNumCols(); w2++,w++)
                    for (int w1=1; w1<=V.GetNumRows(); w1++)
                        V(w1,w2)=Q(w1,w);
                SetV(lr,m,n,V);
            }
        break;
    }
    }
}



void  SiteOperatorImp::ReshapeW(Direction lr,const MatrixRT& Q)
{
    switch (lr)
    {
    case DLeft:
    {
        //  If L has less columns than the Ws then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (Q.GetNumCols()<itsDw.Dw2)
            NewBondDimensions(itsDw.Dw1,Q.GetNumCols(),true);//we must save the old since Q only holds part of W
        //Leave out **first** row and column of W
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT W=GetW(m,n);
                for (int w1=1; w1<=itsDw.Dw1; w1++,w++)
                    for (int w2=1; w2<=itsDw.Dw2; w2++)
                        W(w1,w2)=Q(w,w2);
                SetW(m,n,W);
            }
//        for (int m=0; m<itsd; m++)
//            for (int n=0; n<itsd; n++,w++)
//               cout << "Wnew(" << m << n << ")=" << GetiW(m,n) << endl;
        break;
    }
    case DRight:
    {
        //  If Vdagger has less rows than the Ws then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (Q.GetNumRows()<itsDw.Dw1)
            NewBondDimensions(Q.GetNumRows(),itsDw.Dw2,true);//we must save the old since Q only holds part of W
        //Leave out **last** row and column of W
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT W=GetW(m,n);
                for (int w2=1; w2<=itsDw.Dw2; w2++,w++)
                    for (int w1=1; w1<=itsDw.Dw1; w1++)
                        W(w1,w2)=Q(w1,w);
                SetW(m,n,W);
            }
        break;
    }
    }
}

//
//  For right canonization we need preserve the last row of W.  These are the d, b blocks.
//
void SiteOperatorImp::NewBondDimensions(int D1, int D2, bool saveData)
{
    assert(D1>0);
    assert(D2>0);
    if (itsDw.Dw1==D1 && itsDw.Dw2==D2) return;
//    std::cout << "Reshape from " << itsDw.Dw1 << "," << itsDw.Dw2 << "   to " << D1 << "," << D2 << std::endl;


    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
        {
            VectorRT lastRow=itsWs(m+1,n+1).GetRow(itsDw.Dw1);
            itsWs(m+1,n+1).SetLimits(D1,D2,saveData);
            itsWs(m+1,n+1).GetRow(D1)=lastRow.SubVector(D2);

        }
    itsDw.Dw1=D1;
    itsDw.Dw2=D2;

}




} //namespace
