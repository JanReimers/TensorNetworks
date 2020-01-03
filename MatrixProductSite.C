#include "MatrixProductSite.H"
#include "MPOSite.H"
#include "oml/minmax.h"
#include "oml/cnumeric.h"
#include "oml/vector_io.h"
#include "oml/random.h"
#include <complex>
#include <iostream>

using std::cout;
using std::endl;

MatrixProductSite::MatrixProductSite(int p, int D1, int D2)
    : itsp(p)
    , itsD1(D1)
    , itsD2(D2)
{
    for (int ip=0;ip<itsp;ip++)
    {
        itsAs.push_back(MatrixT(D1,D2));
        Fill(itsAs.back(),std::complex<double>(0.0));
    }
}

MatrixProductSite::~MatrixProductSite()
{
    //dtor
}

MatrixProductSite::Position MatrixProductSite::WhereAreWe() const
{
    return itsAs[0].GetNumRows()==1 ? Left : (itsAs[0].GetNumCols()==1 ? Right : Bulk);
}
//
//  This is a tricker than one might expect.  In particular I can't get the OBC vectors
//  to be both left and right normalized
//
void MatrixProductSite::InitializeWithProductState(int sgn)
{
    if (itsAs[0].GetNumRows()==1)
    {
        int i=1;
        for (pIterT ip=itsAs.begin();ip!=itsAs.end();ip++,i++)
            if (i<=itsD2) (*ip)(1,i)=std::complex<double>(sgn); //Left normalized
    }
    else if (itsAs[0].GetNumCols()==1)
    {
        int i=1;
        for (pIterT ip=itsAs.begin();ip!=itsAs.end();ip++,i++)
            if (i<=itsD1)(*ip)(i,1)=std::complex<double>(sgn);  //Left normalized
    }
    else
    {
        for (pIterT ip=itsAs.begin();ip!=itsAs.end();ip++)
            for (int i=1;i<=Min(itsD1,itsD2);i++)
                (*ip)(i,i)=std::complex<double>(sgn/sqrt(itsp));
    }
}

void MatrixProductSite::InitializeWithRandomState()
{
        for (pIterT ip=itsAs.begin();ip!=itsAs.end();ip++)
            FillRandom(*ip);
}
void MatrixProductSite::SVDLeft_Normalize(VectorT& s, MatrixT& Vdagger)
{
    int N1=s.GetHigh(); //N1=0 on the first site.
    if (N1>0 && N1<itsD1) itsD1=N1; //The contraction below will automatically reshape the As.

    // Where are we in the lattice
    Position lbr=WhereAreWe();
    if (lbr==Bulk)
    {
        for (int in=0; in<itsp; in++)
        {
            MatrixT temp=Contract(s,Vdagger*itsAs[in]);
            itsAs[in].SetLimits(0,0);
            itsAs[in]=temp; //Shallow copy
            assert(itsAs[in].GetNumRows()==itsD1); //Verify shape is correct;
        }
    }
    MatrixT A=ReshapeLeft();
    //
    //  Set up and do SVD
    //
    int N=Min(A.GetNumRows(),A.GetNumCols());
    s.SetLimits(N);
    MatrixT V(N,A.GetNumCols());
    CSVDecomp(A,s,V); //Solves A=U * s * Vdagger  returns V not Vdagger
    Vdagger.SetLimits(0,0);  //Wipe out old data;
    Vdagger=Transpose(conj(V));
    //
    //  Extract As from U
    //
    ReshapeLeft(A);  //A is now U
}

void MatrixProductSite::SVDRightNormalize(MatrixT& U, VectorT& s)
{
    int N1=s.GetHigh(); //N1=0 on the first site.
    if (N1>0 && N1<itsD2) itsD2=N1;

    // Where are we in the lattice
    Position lbr=WhereAreWe();
    if (lbr==Bulk)
    {
        for (int in=0; in<itsp; in++)
        {
            MatrixT temp=Contract(itsAs[in]*U,s);
            itsAs[in].SetLimits(0,0);
            itsAs[in]=temp; //Shallow copy
            assert(itsAs[in].GetNumCols()==itsD2); //Verify shape is correct;
        }
    }
    MatrixT A=ReshapeRight();
    //
    //  Set up and do SVD
    //
    int N=Min(A.GetNumRows(),A.GetNumCols());
    s.SetLimits(N);
    MatrixT V(N,A.GetNumCols());
    CSVDecomp(A,s,V); //Solves A=U * s * Vdagger  returns V not Vdagger
    U.SetLimits(0,0);  //Wipe out old data;
    U=A;
    //
    //  Extract Bs from U
    //
   ReshapeRight(Transpose(conj(V)));  //A is now Vdagger

}

MatrixProductSite::MatrixT  MatrixProductSite::ReshapeLeft()
{
    MatrixT A(itsp*itsD1,itsD2);
    int i2_1=1;
    for (int in=0;in<itsp;in++)
        for (int i1=1;i1<=itsD1;i1++,i2_1++)
            for (int i2=1;i2<=itsD2;i2++)
                A(i2_1,i2)=itsAs[in](i1,i2);
    return A;
}

MatrixProductSite::MatrixT  MatrixProductSite::ReshapeRight()
{
    MatrixT A(itsD1,itsp*itsD2);
    int i2_2=1;
    for (int in=0; in<itsp; in++)
        for (int i2=1; i2<=itsD2; i2++,i2_2++)
            for (int i1=1; i1<=itsD1; i1++)
                A(i1,i2_2)=itsAs[in](i1,i2);
    return A;
}

void MatrixProductSite::Reshape(int D1, int D2, bool saveData)
{
    assert(D1>0);
    assert(D2>0);
    if (itsD1==D1 && itsD2==D2) return;
    itsD1=D1;
    itsD2=D2;
    for (int in=0; in<itsp; in++)
        itsAs[in].SetLimits(itsD1,itsD2,saveData);
}
void MatrixProductSite::ReshapeLeft(const MatrixT& U)
{
    //  If U has less columns than the As then we need to reshape the whole site.
    //  Typically this will happen at the edges of the lattice.
    //
    if (U.GetNumCols()<itsD2) Reshape(itsD1,U.GetNumCols());//This throws away the old data
    int i2_1=1;
    for (int in=0; in<itsp; in++)
        for (int i1=1; i1<=itsD1; i1++,i2_1++)
            for (int i2=1; i2<=itsD2; i2++)
                itsAs[in](i1,i2)=U(i2_1,i2);

}
void MatrixProductSite::ReshapeRight(const MatrixT& Vdagger)
{
    //  If Vdagger has less row than the As then we need to reshape the whole site.
    //  Typically this will happen at the edges of the lattice.
    //
    if (Vdagger.GetNumRows()<itsD1) Reshape(Vdagger.GetNumRows(),itsD2,false);//This throws away the old data
    int i2_2=1;
    for (int in=0; in<itsp; in++)
        for (int i2=1; i2<=itsD2; i2++,i2_2++)
            for (int i1=1; i1<=itsD1; i1++)
                itsAs[in](i1,i2)=Vdagger(i1,i2_2);

}

//
//  Anew(j,i) =  s(j)*VA(j,k)
//
MatrixProductSite::MatrixT MatrixProductSite::Contract(const VectorT& s, const MatrixT& VA)
{
    int N1=VA.GetNumRows();
    int N2=VA.GetNumCols();
    assert(s.GetHigh()==N1);

    MatrixT Anew(N1,N2);
    for(int i2=1; i2<=N2; i2++)
        for(int i1=1; i1<=N1; i1++)
            Anew(i1,i2)=s(i1)*VA(i1,i2);

    return Anew;
}

//
//  Anew(j,i) =  s(j)*VA(j,k)
//
MatrixProductSite::MatrixT MatrixProductSite::Contract(const MatrixT& AU,const VectorT& s)
{
    int N1=AU.GetNumRows();
    int N2=AU.GetNumCols();
    assert(s.GetHigh()==N2);

    MatrixT Anew(N1,N2);
    for(int i2=1; i2<=N2; i2++)
        for(int i1=1; i1<=N1; i1++)
            Anew(i1,i2)=AU(i1,i2)*s(i2);

    return Anew;
}


//
//  Sum_ip A^t(ip) * A(ip)
//
MatrixProductSite::MatrixT MatrixProductSite::GetLeftNorm() const
{
    MatrixT ret(itsD2,itsD2);
    Fill(ret,std::complex<double>(0.0));
    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
    {
        ret+=conj(Transpose((*ip)))*(*ip);
    }
    return ret;
}
//
//  Sum_ip A(ip)*A^t(ip)
//
MatrixProductSite::MatrixT MatrixProductSite::GetRightNorm() const
{
    MatrixT ret(itsD1,itsD1);
    Fill(ret,std::complex<double>(0.0));
    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
        ret+=(*ip)*conj(Transpose((*ip)));
    return ret;
}
//  This is for boundary sites only
//
//  E(1,i,j)=Sum{n,A^t(n;1,i)*A(n;j,1)}=Sum{n,A^*(n;i,1)*A(n;j,1)}
//
MatrixProductSite::MatrixT MatrixProductSite::GetOverlapTransferMatrix() const
{
    assert(GetLimits().GetNumRows()==1 || GetLimits().GetNumCols()==1);

    int D=GetLimits().GetNumRows()==1 ? GetLimits().GetNumCols() : GetLimits().GetNumRows();
    MatrixT ret(D,D);
    Fill(ret,std::complex<double>(0.0));

    if (GetLimits().GetNumRows()==1)
    {
        //Left boundary

        for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
            ret+=conj(Transpose(*ip))*(*ip);
    }
    if (GetLimits().GetNumCols()==1)
    {
        //Right boundary
        for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
            ret+=(*ip)*conj(Transpose(*ip));
    }

    //cout << "OverlapTransferMatrix=" << ret << endl;
    return ret;
}

//
//  N1(n;i,l)=Sum[j,E(a-1,i,j)*A(n;j,l)
//  N2(n;k,l)=Sum(i,A^t(n;i,k)*N1(n;i,l])
//  E(a,k,l)=Sum{n,N2(n;k,l)}
//
MatrixProductSite::MatrixT MatrixProductSite::GetOverlapTransferMatrixLeft(const MatrixT& Em) const
{
    int D=itsAs[0].GetNumCols();
    MatrixT Ea(D,D);
    Fill(Ea,std::complex<double>(0.0));
//    cout << "A lim" <<  itsAs[0].GetLimits() << endl;
//    cout << "Em lim" <<  Em.GetLimits() << endl;
    pVectorT N1s;
    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
    {
        N1s.push_back(Em*(*ip));
        }

    cpIterT iN1=N1s.begin();
    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++,iN1++)
        Ea+=conj(Transpose(*ip))*(*iN1);

    return Ea;
}
MatrixProductSite::MatrixT MatrixProductSite::GetOverlapTransferMatrixRight(const MatrixT& Em) const
{
    int D=itsAs[0].GetNumRows();
    MatrixT Ea(D,D);
    Fill(Ea,std::complex<double>(0.0));
//    cout << "A lim" <<  itsAs[0].GetLimits() << endl;
//    cout << "Em lim" <<  Em.GetLimits() << endl;
    pVectorT N1s;
    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
    {
        N1s.push_back(Em*Transpose(conj(*ip)));
        }

    cpIterT iN1=N1s.begin();
    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++,iN1++)
        Ea+=(*ip)*(*iN1);

    return Ea;
}

MatrixProductSite::MatrixT MatrixProductSite::
GetOverlapMatrix(const MatrixT& Eleft, const MatrixT Eright) const
{
//    cout << "ELeft=" <<  Eleft << endl;
//    cout << "Eright=" <<  Eright << endl;
    MatrixT Sab(itsp*itsD1*itsD2,itsp*itsD1*itsD2);
    int i2_1=1;
    for (int im=0; im<itsp; im++)
        for (int i1=1; i1<=itsD1; i1++)
            for (int i2=1; i2<=itsD2; i2++,i2_1++)
            {
                int i2_2=1;
                for (int in=0; in<itsp; in++)
                    for (int j1=1; j1<=itsD1; j1++)
                        for (int j2=1; j2<=itsD2; j2++, i2_2++)
                            Sab(i2_1,i2_2)=im==in ? Eleft(i1,j1)*Eright(j2,i2) : 0.0;
            }
    return Sab;
}
//
//  Flattened supermatrix indices should be
//  E(i1,j1,k1,i2,j2,k2)=E(k1+D1*(j1+D1*(i1-1)-1),k2+D2*(j2+D2*(i2-1)-1))
//
MatrixProductSite::Matrix6T MatrixProductSite::GetE(const MPOSite* mpos) const
{
    ipairT Dw=mpos->GetDw();
    int Dw1=Dw.first;
    int Dw2=Dw.second;
    Matrix6T E(Dw1,itsD1,Dw2,itsD2,1);

    for (int m=0; m<itsp; m++)
    {
        Matrix4T N=GetN(m,mpos);
        for (int i1=1;i1<=itsD1;i1++)
            for (int j1=1;j1<=itsD1;j1++)
            for (int w1=1;w1<=Dw1;w1++)
            {
                for (int i2=1;i2<=itsD2;i2++)
                for (int j2=1;j2<=itsD2;j2++)
                    for (int w2=1;w2<=Dw2;w2++)
                    {
                        E(w1,i1,j1,w2,i2,j2)+=conj(itsAs[m](i1,i2))*N(w1,j1,w2,j2);
                    }
            }

    }
    //cout << "MPS Elimits=" << E.GetLimits() << endl;
    return E;
}

MatrixProductSite::Matrix4T MatrixProductSite::GetN(int m,const MPOSite* mpos) const
{
    ipairT Dw=mpos->GetDw();
    int Dw1=Dw.first;
    int Dw2=Dw.second;
    Matrix4T N(Dw1,itsD1,Dw2,itsD2);
    N.Fill(std::complex<double>(0.0));
    for (int n=0; n<itsp; n++)
    {
        const MatrixT W=mpos->GetW(n,m);
        for (int w1=1;w1<=Dw1;w1++)
            for (int i1=1;i1<=itsD1;i1++)
                for (int w2=1;w2<=Dw2;w2++)
                    for (int i2=1;i2<=itsD2;i2++)
                    {
 //                       cout << i2_1 << " " << i2_2 << " " << i1 << " " << i2 << " " << j1 << " " << j2 << endl;
                        N(w1,i1,w2,i2)+=W(w1,w2)*itsAs[n](i1,i2);
                    }

    }
    return N;
}

