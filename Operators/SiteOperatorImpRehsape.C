#include "Operators/SiteOperatorImp.H"

namespace TensorNetworks
{


//
//  For LR = {Left/Right} we need to reshape with only {bottom right/top left} portion of matrix which
//  has the intrinsic portion of W.
//  In simple terms we just leave out the {first/last} row and column
//
MatrixRT SiteOperatorImp::ReshapeV(Direction lr) const
{
    MatrixRT A;
    switch (lr)
    {
    case DLeft:
    { //Leave out **first** row and column of W
        A.SetLimits(itsd*itsd*(itsDw.Dw1-1),itsDw.Dw2-1);
        int w=1;
        for (int w1=2; w1<=itsDw.Dw1; w1++)
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,w++)
            {
                const MatrixRT& W=GetiW(m,n);
                for (int w2=2; w2<=itsDw.Dw2; w2++)
                    A(w,w2-1)=W(w1,w2);
            }
        break;
    }
    case DRight:
    { //Leave out **last** row and column of W
        A.SetLimits(itsDw.Dw1-1,itsd*itsd*(itsDw.Dw2-1));
        int w=1;
        for (int w2=1; w2<=itsDw.Dw2-1; w2++)
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,w++)
            {
                const MatrixRT& W=GetiW(m,n);
                for (int w1=1; w1<=itsDw.Dw1-1; w1++)
                    A(w1,w)=W(w1,w2);
            }
        break;
    }
    }
    return A;
}
MatrixRT SiteOperatorImp::ReshapeV1(Direction lr) const
{
    MatrixRT A;
    switch (lr)
    {
    case DLeft:
    { //Leave out **first** row and column
        A.SetLimits(itsd*itsd*(itsDw.Dw1-1),itsDw.Dw2-1);
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                const MatrixRT& W=GetiW(m,n);
                for (int w1=2; w1<=itsDw.Dw1; w1++)
                {

                    for (int w2=2; w2<=itsDw.Dw2; w2++)
                        A(w,w2-1)=W(w1,w2);
                    w++;
                }
            }
        break;
    }
    case DRight:
    { //Leave out **last** row and column
        A.SetLimits(itsDw.Dw1-1,itsd*itsd*(itsDw.Dw2-1));
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                const MatrixRT& W=GetiW(m,n);
                for (int w2=1; w2<=itsDw.Dw2-1; w2++,w++)
                   for (int w1=1; w1<=itsDw.Dw1-1; w1++)
                        A(w1,w)=W(w1,w2);
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
            Reshape(itsDw.Dw1,Q.GetNumCols()+1,true);//we must save the old since Q only holds part of W
        //Leave out **first** row and column of W
        int w=1;
        for (int w1=2; w1<=itsDw.Dw1; w1++)
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,w++)
            {
                MatrixRT W=GetiW(m,n);
                for (int w2=2; w2<=itsDw.Dw2; w2++)
                    W(w1,w2)=Q(w,w2-1);
                SetiW(m,n,W);
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
        if (Q.GetNumRows()+1<itsDw.Dw1)
            Reshape(Q.GetNumRows()+1,itsDw.Dw2,true);//we must save the old since Q only holds part of W
        //Leave out **last** row and column of W
        int w=1;
        for (int w2=1; w2<=itsDw.Dw2-1; w2++)
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++,w++)
            {
                MatrixRT W=GetiW(m,n);
                for (int w1=1; w1<=itsDw.Dw1-1; w1++)
                    W(w1,w2)=Q(w1,w);
                SetiW(m,n,W);
            }
        break;
    }
    }
    Init_lr();
}
void  SiteOperatorImp::ReshapeV1(Direction lr,const MatrixRT& Q)
{
    switch (lr)
    {
    case DLeft:
    {
        //  If L has less columns than the Ws then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (Q.GetNumCols()+1<itsDw.Dw2) Reshape(itsDw.Dw1,Q.GetNumCols()+1);//This throws away the old data
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT W=GetiW(m,n);
                for (int w1=2; w1<=itsDw.Dw1; w1++,w++)
                    for (int w2=2; w2<=itsDw.Dw2; w2++)
                        W(w1,w2)=Q(w,w2-1);
                SetiW(m,n,W);
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
        if (Q.GetNumRows()+1<itsDw.Dw1) Reshape(Q.GetNumRows(),itsDw.Dw2+1,false);//This throws away the old data
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT W=GetiW(m,n);
                for (int w2=1; w2<=itsDw.Dw2-1; w2++,w++)
                    for (int w1=1; w1<=itsDw.Dw1-1; w1++)
                        W(w1,w2)=Q(w1,w);
                SetiW(m,n,W);
            }
        break;
    }
    }
    Init_lr();
}


MatrixRT SiteOperatorImp::Reshape(Direction lr) const
{
    MatrixRT A;
    switch (lr)
    {
    case DLeft:
    {
        A.SetLimits(itsd*itsd*itsDw.Dw1,itsDw.Dw2);
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                const MatrixRT& W=GetiW(m,n);
                for (int w1=1; w1<=itsDw.Dw1; w1++,w++)
                    for (int w2=1; w2<=itsDw.Dw2; w2++)
                        A(w,w2)=W(w1,w2);
            }
        break;
    }
    case DRight:
    {
        A.SetLimits(itsDw.Dw1,itsd*itsd*itsDw.Dw2);
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                const MatrixRT& W=GetiW(m,n);
                for (int w2=1; w2<=itsDw.Dw2; w2++,w++)
                   for (int w1=1; w1<=itsDw.Dw1; w1++)
                        A(w1,w)=W(w1,w2);
            }
        break;
    }
    }
    return A;
}

void SiteOperatorImp::Reshape(int D1, int D2, bool saveData)
{
    assert(D1>0);
    assert(D2>0);
    if (itsDw.Dw1==D1 && itsDw.Dw2==D2) return;
//    cout << "Reshape from " << itsDwBulk.Dw1 << "," << itsDwBulk.Dw2 << "   to " << D1 << "," << D2 << endl;
    itsDw.Dw1=D1;
    itsDw.Dw2=D2;
    for (int m=0; m<itsd; m++)
        for (int n=0; n<itsd; n++)
            itsWs(m+1,n+1).SetLimits(itsDw.Dw1,itsDw.Dw2,saveData);

    isShapeDirty=true;
}

void  SiteOperatorImp::Reshape(Direction lr,const MatrixRT& UV)
{
    switch (lr)
    {
    case DLeft:
    {
        //  If U has less columns than the Ws then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (UV.GetNumCols()<itsDw.Dw2) Reshape(itsDw.Dw1,UV.GetNumCols());//This throws away the old data
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT W=GetiW(m,n);
                for (int w1=1; w1<=itsDw.Dw1; w1++,w++)
                    for (int w2=1; w2<=itsDw.Dw2; w2++)
                        W(w1,w2)=UV(w,w2);
                SetiW(m,n,W);
            }
        break;
    }
    case DRight:
    {
        //  If Vdagger has less rows than the Ws then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (UV.GetNumRows()<itsDw.Dw1) Reshape(UV.GetNumRows(),itsDw.Dw2,false);//This throws away the old data
        int w=1;
        for (int m=0; m<itsd; m++)
            for (int n=0; n<itsd; n++)
            {
                MatrixRT W=GetiW(m,n);
                for (int w2=1; w2<=itsDw.Dw2; w2++,w++)
                    for (int w1=1; w1<=itsDw.Dw1; w1++)
                        W(w1,w2)=UV(w1,w);
                SetiW(m,n,W);
            }
        break;
    }
    }
    Init_lr();
}




} //namespace
