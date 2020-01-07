#include "MatrixProductSite.H"
#include "MPOSite.H"
#include "Vector3.H"
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
    , itsNumUpdates(0)
    , itsBondEntropy(0.0)
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
void MatrixProductSite::InitializeWith(State state,int sgn)
{
    switch (state)
    {
    case Product :
        {
            Position lbr=WhereAreWe();
            switch(lbr)
            {
            case  Left :
                {
                    int i=1;
                    for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++,i++)
                        if (i<=itsD2)
                            (*ip)(1,i)=std::complex<double>(sgn); //Left normalized
                    break;
                }
            case Right :
                {
                    int i=1;
                    for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++,i++)
                        if (i<=itsD1)
                            (*ip)(i,1)=std::complex<double>(sgn);  //Left normalized
                    break;
                }
            case Bulk :
                {
                    for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
                        for (int i=1; i<=Min(itsD1,itsD2); i++)
                            (*ip)(i,i)=std::complex<double>(sgn/sqrt(itsp));
                    break;
                }
            }
            break;
        }
    case Random :
        {
            for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
                FillRandom(*ip);
            break;
        }
    case Neel :
        {
            for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
                Fill(*ip,eType(0.0));
            if (sgn== 1)
                itsAs[0     ](1,1)=1.0;
            if (sgn==-1)
                itsAs[itsp-1](1,1)=1.0;

            break;
        }
    }
}

void MatrixProductSite::SVDLeft_Normalize(VectorT& s, MatrixT& Vdagger)
{

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
    itsBondEntropy=CalcBondEntropy(s);
//    cout << "Bond S=" << itsBondEntropy << endl;
}

void MatrixProductSite::SVDRightNormalize(MatrixT& U, VectorT& s)
{
    MatrixT A=ReshapeRight();
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
    itsBondEntropy=CalcBondEntropy(s);
//    cout << "Bond S=" << itsBondEntropy << endl;
}

void MatrixProductSite::ReshapeFromLeft (int D1)
{
    Reshape(   D1,itsD2,true);
}

void MatrixProductSite::ReshapeFromRight(int D2)
{
    Reshape(itsD1,   D2,true);
}

void MatrixProductSite::Rescale(double norm)
{
    for (int n=0;n<itsp;n++) itsAs[n]/=norm;
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
MatrixProductSite::MatrixT MatrixProductSite::Contract1(const VectorT& s, const MatrixT& VA)
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
MatrixProductSite::MatrixT MatrixProductSite::Contract1(const MatrixT& AU,const VectorT& s)
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

std::string MatrixProductSite::GetNormStatus() const
{
    std::string ret;
    if (IsLeftNormalized()) ret="A";
    else if (IsRightNormalized()) ret="B";
    else ret="M";
    ret+=std::to_string(itsNumUpdates);
    return ret;
}

bool MatrixProductSite::IsLeftNormalized() const
{
    return IsUnit(GetLeftNorm(),1e-12);
}
bool MatrixProductSite::IsRightNormalized() const
{
    return IsUnit(GetRightNorm(),1e-12);
}

bool MatrixProductSite::IsUnit(const MatrixT& m,double eps)
{
    assert(m.GetNumRows()==m.GetNumCols());
    int N=m.GetNumRows();
    MatrixT I(N,N);
    Unit(I);
    return Max(abs(m-I))<eps;
}

double MatrixProductSite::CalcBondEntropy(const VectorT& s)
{
    int N=s.size();
    double ret=0.0;
    for (int i=1;i<=N;i++)
    {
        double s2=s(i)*s(i);
        if (s2>0.0) ret-=s2*log(s2);
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
MatrixProductSite::MatrixT MatrixProductSite::GetE() const
{
    assert(GetLimits().GetNumRows()==1 || GetLimits().GetNumCols()==1);

    int D=GetLimits().GetNumRows()==1 ? GetLimits().GetNumCols() : GetLimits().GetNumRows();
    MatrixT E(D,D);
    Fill(E,std::complex<double>(0.0));

    if (GetLimits().GetNumRows()==1)
    {
        //Left boundary

        for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
            E+=conj(Transpose(*ip))*(*ip);
    }
    if (GetLimits().GetNumCols()==1)
    {
        //Right boundary
        for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
            E+=(*ip)*conj(Transpose(*ip));
    }

    //cout << "OverlapTransferMatrix=" << E << endl;
    return E;
}

//
//  N1(n;i,l)=Sum[j,E(a-1,i,j)*A(n;j,l)
//  N2(n;k,l)=Sum(i,A^t(n;i,k)*N1(n;i,l])
//  E(a,k,l)=Sum{n,N2(n;k,l)}
//
MatrixProductSite::MatrixT MatrixProductSite::GetELeft(const MatrixT& Em) const
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
MatrixProductSite::MatrixT MatrixProductSite::GetERight(const MatrixT& Em) const
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

// TODO (jan#1#): Use Matrix4 for the N
MatrixProductSite::MatrixT MatrixProductSite::
GetNeff(const MatrixT& Eleft, const MatrixT Eright) const
{
//    cout << "ELeft=" <<  Eleft << endl;
//    cout << "Eright=" <<  Eright << endl;
    MatrixT Neff(itsp*itsD1*itsD2,itsp*itsD1*itsD2);
    int i2_1=1;
    for (int im=0; im<itsp; im++)
        for (int i1=1; i1<=itsD1; i1++)
            for (int i2=1; i2<=itsD2; i2++,i2_1++)
            {
                int i2_2=1;
                for (int in=0; in<itsp; in++)
                    for (int j1=1; j1<=itsD1; j1++)
                        for (int j2=1; j2<=itsD2; j2++, i2_2++)
                            Neff(i2_1,i2_2)=im==in ? Eleft(i1,j1)*Eright(j2,i2) : 0.0;
            }
    return Neff;
}
//
//  Operator transfer supermatrix
//
MatrixProductSite::Matrix6T MatrixProductSite::GetEO(const MPOSite* mpos) const
{
    ipairT Dw=mpos->GetDw();
    int Dw1=Dw.first;
    int Dw2=Dw.second;
    Matrix6T EO(Dw1,itsD1,itsD1,Dw2,itsD2,itsD2,1);

    for (int m=0; m<itsp; m++)
    for (int n=0; n<itsp; n++)
    {
        const MatrixT W=mpos->GetW(m,n);
 //       cout << "W(" << m << "," << n << ")=" << W << endl;
//        Matrix4T NO=GetNO(m,mpos);
        for (int i1=1;i1<=itsD1;i1++)
            for (int j1=1;j1<=itsD1;j1++)
            for (int w1=1;w1<=Dw1;w1++)
            {
                for (int i2=1;i2<=itsD2;i2++)
                for (int j2=1;j2<=itsD2;j2++)
                    for (int w2=1;w2<=Dw2;w2++)
                    {
                        EO(w1,i1,j1,w2,i2,j2)+=conj(itsAs[m](i1,i2))*W(w1,w2)*itsAs[n](j1,j2);
                    }
            }

    }
    //cout << "MPS Elimits=" << E.GetLimits() << endl;
    return EO;
}

MatrixProductSite::Matrix4T MatrixProductSite::GetNO(int m,const MPOSite* mpos) const
{
    ipairT Dw=mpos->GetDw();
    int Dw1=Dw.first;
    int Dw2=Dw.second;
    Matrix4T NO(Dw1,itsD1,Dw2,itsD2);
    NO.Fill(std::complex<double>(0.0));
    for (int n=0; n<itsp; n++)
    {
        const MatrixT W=mpos->GetW(n,m);
        for (int w1=1;w1<=Dw1;w1++)
            for (int i1=1;i1<=itsD1;i1++)
                for (int w2=1;w2<=Dw2;w2++)
                    for (int i2=1;i2<=itsD2;i2++)
                    {
 //                       cout << i2_1 << " " << i2_2 << " " << i1 << " " << i2 << " " << j1 << " " << j2 << endl;
                        NO(w1,i1,w2,i2)+=W(w1,w2)*itsAs[n](i1,i2);
                    }

    }
    return NO;
}

MatrixProductSite::Vector3T MatrixProductSite::IterateLeft_F(const MPOSite* mpos, const Vector3T& Fam1)
{
    ipairT Dw=mpos->GetDw();
    int Dw2=Dw.second;
    Vector3T F(Dw2,itsD2,itsD2,1);
    for (int w2=1;w2<=Dw2;w2++)
        for (int i2=1;i2<=itsD2;i2++)
            for (int j2=1;j2<=itsD2;j2++)
                F(w2,i2,j2)=ContractAWFA(w2,i2,j2,mpos,Fam1);
    return F;
}

MatrixProductSite::eType MatrixProductSite::ContractAWFA(int w2, int i2, int j2, const MPOSite* mpos, const Vector3T& Fam1)
{
    eType awfa(0.0);
     for (int m=0; m<itsp; m++)
        for (int i1=1;i1<=itsD1;i1++)
            awfa+=conj(itsAs[m](i1,i2))*ContractWFA(m,w2,i1,j2,mpos,Fam1);

    return awfa;
}

MatrixProductSite::eType MatrixProductSite::ContractWFA(int m, int w2, int i1, int j2, const MPOSite* mpos, const Vector3T& Fam1)
{
    eType wfa(0.0);
     for (int n=0; n<itsp; n++)
     {
        MatrixT Wmn=mpos->GetW(m,n);
        int Dw1=Wmn.GetNumRows();
        for (int w1=1;w1<=Dw1;w1++)
            wfa+=Wmn(w1,w2)*ContractFA(n,w1,i1,j2,Fam1);
    }
    return wfa;
}

MatrixProductSite::eType MatrixProductSite::ContractFA(int n, int w1, int i1, int j2, const Vector3T& Fam1)
{
    eType fa(0.0);
    for (int j1=1;j1<=itsD1;j1++)
        fa+=Fam1(w1,i1,j1)*itsAs[n](j1,j2);
    return fa;
}

MatrixProductSite::Vector3T MatrixProductSite::IterateRightF(const MPOSite* mpos, const Vector3T& Fap1)
{
    ipairT Dw=mpos->GetDw();
    int Dw1=Dw.first;
    Vector3T F(Dw1,itsD1,itsD1,1);
    for (int w1=1;w1<=Dw1;w1++)
        for (int i1=1;i1<=itsD1;i1++)
            for (int j1=1;j1<=itsD1;j1++)
                F(w1,i1,j1)=ContractBWFB(w1,i1,j1,mpos,Fap1);
    return F;
}

MatrixProductSite::eType MatrixProductSite::ContractBWFB(int w1, int i1, int j1, const MPOSite* mpos, const Vector3T& Fap1)
{
    eType bwfb(0.0);
     for (int m=0; m<itsp; m++)
        for (int i2=1;i2<=itsD2;i2++)
            bwfb+=conj(itsAs[m](i1,i2))*ContractWFB(m,w1,i2,j1,mpos,Fap1);

    return bwfb;
}

MatrixProductSite::eType MatrixProductSite::ContractWFB(int m, int w1, int i2, int j1, const MPOSite* mpos, const Vector3T& Fap1)
{
    eType wfb(0.0);
     for (int n=0; n<itsp; n++)
     {
        MatrixT Wmn=mpos->GetW(m,n);
        int Dw2=Wmn.GetNumCols();
        for (int w2=1;w2<=Dw2;w2++)
            wfb+=Wmn(w1,w2)*ContractFB(n,w2,i2,j1,Fap1);
    }
    return wfb;
}

MatrixProductSite::eType MatrixProductSite::ContractFB(int n, int w2, int i2, int j1, const Vector3T& Fap1)
{
    eType fb(0.0);
    for (int j2=1;j2<=itsD2;j2++)
        fb+=Fap1(w2,i2,j2)*itsAs[n](j1,j2);
    return fb;
}


double MatrixProductSite::ContractHeff(const Matrix6T& Heff) const
{
    eType E(0.0);
    for (int m=0; m<itsp; m++)
        for (int n=0; n<itsp; n++)
            for (int i1=1; i1<=itsD1; i1++)
                for (int j1=1; j1<=itsD1; j1++)
                    for (int i2=1; i2<=itsD2; i2++)
                        for (int j2=1; j2<=itsD2; j2++)
                        {
                            E+=conj(itsAs[m](i1,i2))*Heff(m,i1,i2,n,j1,j2)*itsAs[n](j1,j2);
                        }

    //cout << "fabs(std::imag(E))" <<  fabs(std::imag(E)) << endl;
    double iE=fabs(std::imag(E));
    if (iE>1e-8)
        cout << "Warning ContractHeff imag(E)=" << iE << endl;
    return real(E);
}
double MatrixProductSite::ContractHeff(const MatrixT& Heff) const
{
    Vector3<eType> As(itsp,itsD1,itsD2);
    for (int m=0; m<itsp; m++)
        for (int i1=1; i1<=itsD1; i1++)
            for (int i2=1; i2<=itsD2; i2++)
                As(m,i1,i2)=itsAs[m](i1,i2);

    VectorCT AsFlat=As.Flatten();
    eType E=conj(AsFlat)*Heff*AsFlat;
    double iE=fabs(std::imag(E));
    if (iE>1e-8)
        cout << "Warning ContractFlattenedHeff imag(E)=" << iE << endl;
    return real(E);
}

void MatrixProductSite::Update(const VectorCT& newAs)
{
    Vector3<eType> As(itsp,itsD1,itsD2,newAs); //Unflatten
    for (int m=0; m<itsp; m++)
    {
 //       cout << "before A[" << m << "]=" << itsAs[m] << endl;
        for (int i1=1; i1<=itsD1; i1++)
            for (int i2=1; i2<=itsD2; i2++)
                itsAs[m](i1,i2)=As(m,i1,i2);
//       cout << "after  A[" << m << "]=" << itsAs[m] << endl;
    }
    itsNumUpdates++;
}



void MatrixProductSite::Contract(const VectorT& s, const MatrixT& Vdagger)
{
    int N1=s.GetHigh(); //N1=0 on the first site.
    if (N1>0 && N1<itsD1) itsD1=N1; //The contraction below will automatically reshape the As.

    for (int in=0; in<itsp; in++)
    {
        MatrixT temp=Contract1(s,Vdagger*itsAs[in]);
        itsAs[in].SetLimits(0,0);
        itsAs[in]=temp; //Shallow copy
//        cout << "A[" << in << "]=" << itsAs[in] << endl;
        assert(itsAs[in].GetNumRows()==itsD1); //Verify shape is correct;
    }
}

void MatrixProductSite::Contract(const MatrixT& U, const VectorT& s)
{
    int N1=s.GetHigh(); //N1=0 on the first site.
    if (N1>0 && N1<itsD2)
        itsD2=N1; //The contraction below will automatically reshape the As.
    for (int in=0; in<itsp; in++)
    {
        MatrixT temp=Contract1(itsAs[in]*U,s);
        itsAs[in].SetLimits(0,0);
        itsAs[in]=temp; //Shallow copy
        assert(itsAs[in].GetNumCols()==itsD2); //Verify shape is correct;
    }
}
