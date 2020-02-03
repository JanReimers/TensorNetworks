#include "TensorNetworksImp/MatrixProductSite.H"
#include "TensorNetworksImp/Bond.H"
#include "TensorNetworks/SiteOperator.H"
#include "TensorNetworks/Dw12.H"
#include "oml/minmax.h"
#include "oml/cnumeric.h"
#include "oml/vector_io.h"
#include "oml/random.h"
#include <iostream>

using std::cout;
using std::endl;

MatrixProductSite::MatrixProductSite(TensorNetworks::Position lbr, Bond* leftBond, Bond* rightBond,int p, int D1, int D2)
    : itsLeft_Bond(leftBond)
    , itsRightBond(rightBond)
    , itsp(p)
    , itsD1(D1)
    , itsD2(D2)
    , itsHLeft_Cache(1,1,1,1)
    , itsHRightCache(1,1,1,1)
    , itsEigenSolver(1e-12)
    , itsNumUpdates(0)
    , itsHeffDensity(0)
    , itsEmin(0.0)
    , itsGapE(0.0)
    , itsIterDE(1.0)
    , itsPosition(lbr)
{
    if (lbr==TensorNetworks::Left)
    {
        assert(itsRightBond);
    }
    if (lbr==TensorNetworks::Right)
    {
        assert(itsLeft_Bond);
    }

    for (int ip=0;ip<itsp;ip++)
    {
        itsAs.push_back(MatrixCT(D1,D2));
        Fill(itsAs.back(),std::complex<double>(0.0));
    }
    itsHLeft_Cache(1,1,1)=1.0;
    itsHRightCache(1,1,1)=1.0;
}

MatrixProductSite::~MatrixProductSite()
{
    //dtor
}

//
//  This is a tricker than one might expect.  In particular I can't get the OBC vectors
//  to be both left and right normalized
//
void MatrixProductSite::InitializeWith(TensorNetworks::State state,int sgn)
{
    switch (state)
    {
    case TensorNetworks::Product :
        {
            TensorNetworks::Position lbr=WhereAreWe();
            switch(lbr)
            {
            case  TensorNetworks::Left :
                {
                    int i=1;
                    for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++,i++)
                        if (i<=itsD2)
                            (*ip)(1,i)=std::complex<double>(sgn); //Left normalized
                    break;
                }
            case TensorNetworks::Right :
                {
                    int i=1;
                    for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++,i++)
                        if (i<=itsD1)
                            (*ip)(i,1)=std::complex<double>(sgn);  //Left normalized
                    break;
                }
            case TensorNetworks::Bulk :
                {
                    for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
                        for (int i=1; i<=Min(itsD1,itsD2); i++)
                            (*ip)(i,i)=std::complex<double>(sgn/sqrt(itsp));
                    break;
                }
            }
            break;
        }
    case TensorNetworks::Random :
        {
            for (pIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
                FillRandom(*ip);
            break;
        }
    case TensorNetworks::Neel :
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

void MatrixProductSite::SVDLeft_Normalize(VectorT& s, MatrixCT& Vdagger)
{

    MatrixCT A=ReshapeLeft();
    //
    //  Set up and do SVD
    //
    int N=Min(A.GetNumRows(),A.GetNumCols());
    s.SetLimits(N);
    MatrixCT V(N,A.GetNumCols());
    CSVDecomp(A,s,V); //Solves A=U * s * Vdagger  returns V not Vdagger
    Vdagger.SetLimits(0,0);  //Wipe out old data;
    Vdagger=Transpose(conj(V));
    //
    //  Extract As from U
    //
    ReshapeLeft(A);  //A is now U
    if (itsRightBond) itsRightBond->SetSingularValues(s);
}

void MatrixProductSite::SVDRightNormalize(MatrixCT& U, VectorT& s)
{
    MatrixCT A=ReshapeRight();
    int N=Min(A.GetNumRows(),A.GetNumCols());
    s.SetLimits(N);
    MatrixCT V(N,A.GetNumCols());
    CSVDecomp(A,s,V); //Solves A=U * s * Vdagger  returns V not Vdagger
    U.SetLimits(0,0);  //Wipe out old data;
    U=A;
    //
    //  Extract Bs from U
    //
    ReshapeRight(Transpose(conj(V)));  //A is now Vdagger
    assert(itsLeft_Bond);
    if (itsLeft_Bond) itsLeft_Bond->SetSingularValues(s);
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

MatrixProductSite::MatrixCT  MatrixProductSite::ReshapeLeft()
{
    MatrixCT A(itsp*itsD1,itsD2);
    int i2_1=1;
    for (int in=0;in<itsp;in++)
        for (int i1=1;i1<=itsD1;i1++,i2_1++)
            for (int i2=1;i2<=itsD2;i2++)
                A(i2_1,i2)=itsAs[in](i1,i2);
    return A;
}

MatrixProductSite::MatrixCT  MatrixProductSite::ReshapeRight()
{
    MatrixCT A(itsD1,itsp*itsD2);
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
void MatrixProductSite::ReshapeLeft(const MatrixCT& U)
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
void MatrixProductSite::ReshapeRight(const MatrixCT& Vdagger)
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
MatrixProductSite::MatrixCT MatrixProductSite::Contract1(const VectorT& s, const MatrixCT& VA)
{
    int N1=VA.GetNumRows();
    int N2=VA.GetNumCols();
    assert(s.GetHigh()==N1);

    MatrixCT Anew(N1,N2);
    for(int i2=1; i2<=N2; i2++)
        for(int i1=1; i1<=N1; i1++)
            Anew(i1,i2)=s(i1)*VA(i1,i2);

    return Anew;
}

//
//  Anew(j,i) =  s(j)*VA(j,k)
//
MatrixProductSite::MatrixCT MatrixProductSite::Contract1(const MatrixCT& AU,const VectorT& s)
{
    int N1=AU.GetNumRows();
    int N2=AU.GetNumCols();
    assert(s.GetHigh()==N2);

    MatrixCT Anew(N1,N2);
    for(int i2=1; i2<=N2; i2++)
        for(int i1=1; i1<=N1; i1++)
            Anew(i1,i2)=AU(i1,i2)*s(i2);

    return Anew;
}


//
//  Sum_ip A^t(ip) * A(ip)
//
MatrixProductSite::MatrixCT MatrixProductSite::GetLeftNorm() const
{
    MatrixCT ret(itsD2,itsD2);
    Fill(ret,std::complex<double>(0.0));
    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
    {
        ret+=conj(Transpose((*ip)))*(*ip);
    }
    return ret;
}

std::string MatrixProductSite::GetNormStatus(double eps) const
{
//    StreamableObject::SetToPretty();
//    for (int ip=0;ip<itsp; ip++)
//        cout << "A[" << ip << "]=" << itsAs[ip] << endl;
    std::string ret;
    if (IsLeftNormalized(eps))
    {
        if (IsRightNormalized(eps))
            ret="I"; //This should be rare
        else
            ret="A";
    }
    else
        if (IsRightNormalized(eps))
            ret="B";
        else
            ret="M";

    ret+=std::to_string(itsNumUpdates);
    return ret;
}

void MatrixProductSite::Report(std::ostream& os) const
{
    os << std::setprecision(3)
    << std::setw(4) << itsD1
    << std::setw(4)  << itsD2 << std::fixed
    << std::setw(5)  << itsNumUpdates << "      "
    << std::setw(5)  << itsHeffDensity << "   " << std::setprecision(7)
    << std::setw(9)  << itsEmin << "     " << std::setprecision(4)
    << std::setw(5)  << itsGapE << "   " << std::scientific
    << std::setw(5)  << itsIterDE << "  "
    ;
}

double MatrixProductSite::GetMaxAmplitude() const
{
    double ret=0.0;
    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
    {
        double aa=Max(abs(*ip));
        if (aa>ret) ret=aa;
    }
    return ret;
}

bool MatrixProductSite::IsLeftNormalized(double eps) const
{
    return IsUnit(GetLeftNorm(),eps);
}
bool MatrixProductSite::IsRightNormalized(double eps) const
{
    return IsUnit(GetRightNorm(),eps);
}

bool MatrixProductSite::IsUnit(const MatrixCT& m,double eps)
{
    assert(m.GetNumRows()==m.GetNumCols());
    int N=m.GetNumRows();
    MatrixCT I(N,N);
    Unit(I);
    return Max(abs(m-I))<eps;
}

//
//  Sum_ip A(ip)*A^t(ip)
//
MatrixProductSite::MatrixCT MatrixProductSite::GetRightNorm() const
{
    MatrixCT ret(itsD1,itsD1);
    Fill(ret,std::complex<double>(0.0));
    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
        ret+=(*ip)*conj(Transpose((*ip)));
    return ret;
}
//  This is for boundary sites only
//
//  E(1,i,j)=Sum{n,A^t(n;1,i)*A(n;j,1)}=Sum{n,A^*(n;i,1)*A(n;j,1)}
//
MatrixProductSite::MatrixCT MatrixProductSite::GetE() const
{
    assert(WhereAreWe()!=TensorNetworks::Bulk);
    int D= (WhereAreWe()==TensorNetworks::Left) ? GetLimits().GetNumCols() : GetLimits().GetNumRows();
    MatrixCT E(D,D);
    Fill(E,std::complex<double>(0.0));

    if (WhereAreWe()==TensorNetworks::Left)
    {
        for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
            E+=conj(Transpose(*ip))*(*ip);
    }
    if (WhereAreWe()==TensorNetworks::Right)
    {
        for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
            E+=(*ip)*conj(Transpose(*ip));
    }
    return E;
}

//
//  N1(n;i,l)=Sum[j,E(a-1,i,j)*A(n;j,l)
//  N2(n;k,l)=Sum(i,A^t(n;i,k)*N1(n;i,l])
//  E(a,k,l)=Sum{n,N2(n;k,l)}
//
MatrixProductSite::MatrixCT MatrixProductSite::GetELeft(const MatrixCT& Em) const
{
    int D=itsAs[0].GetNumCols();
    MatrixCT Ea(D,D);
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
MatrixProductSite::MatrixCT MatrixProductSite::GetERight(const MatrixCT& Em) const
{
    int D=itsAs[0].GetNumRows();
    MatrixCT Ea(D,D);
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
MatrixProductSite::MatrixCT MatrixProductSite::
GetNeff(const MatrixCT& Eleft, const MatrixCT Eright) const
{
//    cout << "ELeft=" <<  Eleft << endl;
//    cout << "Eright=" <<  Eright << endl;
    MatrixCT Neff(itsp*itsD1*itsD2,itsp*itsD1*itsD2);
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
MatrixProductSite::Matrix6T MatrixProductSite::GetEO(const SiteOperator* so) const
{
    TensorNetworks::ipairT Dw=so->GetDw();
    int Dw1=Dw.first;
    int Dw2=Dw.second;
    Matrix6T EO(Dw1,itsD1,itsD1,Dw2,itsD2,itsD2,1);

    for (int m=0; m<itsp; m++)
    for (int n=0; n<itsp; n++)
    {
        const MatrixT& W=so->GetW(m,n);
 //       cout << "W(" << m << "," << n << ")=" << W << endl;
//        Matrix4T NO=GetNO(m,so);
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

MatrixProductSite::Matrix4T MatrixProductSite::GetNO(int m,const SiteOperator* so) const
{
    TensorNetworks::ipairT Dw=so->GetDw();
    int Dw1=Dw.first;
    int Dw2=Dw.second;
    Matrix4T NO(Dw1,itsD1,Dw2,itsD2);
    NO.Fill(std::complex<double>(0.0));
    for (int n=0; n<itsp; n++)
    {
        const MatrixT& W=so->GetW(n,m);
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

MatrixProductSite::Matrix6T MatrixProductSite::
GetHeff(const SiteOperator* mops,const Vector3T& L,const Vector3T& R) const
{

#ifdef DEBUG3
    Vector3T L=GetEOLeft_Iterate(mpo,isite,false);
    Vector3T R=GetEORightIterate(mpo,isite,false);
    cout << "L.Flatten()     =" << L.Flatten()<< endl;
    cout << "Lcache.Flatten()=" << Lcache.Flatten()<< endl;
    double errorL=Max(abs(L.Flatten()-Lcache.Flatten()));
//    cout << "R.Flatten()     =" << R.Flatten()<< endl;
//    cout << "Rcache.Flatten()=" << Rcache.Flatten()<< endl;
    double errorR=Max(abs(R.Flatten()-Rcache.Flatten()));
    if (errorL>eps || errorR>eps)
    {
        cout << "Warning Heff errors Left,Rigt=" << std::scientific << errorL << " " << errorR << endl;
        cout << "L=" << L  << endl;
        cout << "Lcache(ia-1)=" << Lcache  << endl;
        cout << "Lcache(ia  )=" << GetHLeft_Cache(isite)  << endl;
        cout << "R=" << R  << endl;
        cout << "Rcache(ia+1)=" << Rcache  << endl;
    }
    assert(errorL<eps);
    assert(errorR<eps);

#endif

    assert(mops);
    Matrix6<eType> Heff(itsp,itsD1,itsD2,itsp,itsD1,itsD2);
    const Dw12& Dws=mops->GetDw12();

    for (int m=0; m<itsp; m++)
        for (int n=0; n<itsp; n++)
        {
            const MatrixT& W=mops->GetW(m,n);
            assert(W.GetNumRows()==Dws.Dw1);
            Vector3T WR(Dws.Dw1,itsD2,itsD2,1);
            for (int w1=1; w1<=Dws.Dw1; w1++)
                for (int i2=1; i2<=itsD2; i2++)
                    for (int j2=1; j2<=itsD2; j2++)
                        WR(w1,i2,j2)=ContractWR(w1,i2,j2,W,Dws.w2_last(w1),R);

            for (int i1=1; i1<=itsD1; i1++)
                for (int j1=1; j1<=itsD1; j1++)
                    for (int i2=1; i2<=itsD2; i2++)
                        for (int j2=1; j2<=itsD2; j2++)
                        {
                            eType LWR(0.0);
                            for (int w1=1; w1<=W.GetNumRows(); w1++)
                                LWR+=L(w1,i1,j1)*WR(w1,i2,j2);
                            Heff(m,i1,i2,n,j1,j2)=LWR;
                        }
        }

    return Heff;

}

eType MatrixProductSite::
ContractWR(int w1, int i2, int j2,const MatrixT& W, int Dw2,const Vector3T& R) const
{
    eType WR(0.0);
    for (int w2=1; w2<=Dw2; w2++)
        if (W(w1,w2)!=0.0)
            WR+=W(w1,w2)*R(w2,i2,j2);
    return WR;
}


MatrixProductSite::Matrix6T MatrixProductSite::
GetHeff(const SiteOperator* mops,const Matrix6T& L,const Matrix6T& R) const
{
    assert(mops);
    Matrix6<eType> Heff(itsp,itsD1,itsD2,itsp,itsD1,itsD2);

    for (int m=0; m<itsp; m++)
        for (int i1=1; i1<=itsD1; i1++)
            for (int j1=1; j1<=itsD1; j1++)
            {
                for (int n=0; n<itsp; n++)
                {
                    const MatrixT& W=mops->GetW(m,n);
                    for (int i2=1; i2<=itsD2; i2++)
                        for (int j2=1; j2<=itsD2; j2++)
                        {
                            eType temp(0.0);
                            for (int w1=1; w1<=W.GetNumRows(); w1++)
                                for (int w2=1; w2<=W.GetNumCols(); w2++)
                                {
                                    temp+=L(1,1,1,w1,i1,j1)*W(w1,w2)*R(w2,i2,j2,1,1,1);
                                }

                            Heff(m,i1,i2,n,j1,j2)=temp;
                        }
                }
            }
    return Heff;

}

void MatrixProductSite::Refine(const MatrixCT& Heff,const Epsilons& eps)
{
    assert(Heff.GetNumRows()==Heff.GetNumCols());
    int N=Heff.GetNumRows();
    Vector<double>  eigenValues(N);
    itsEigenSolver.Solve(Heff,2,eps); //Get lowest two eigen values/states

    eigenValues=itsEigenSolver.GetEigenValues();

    itsIterDE=eigenValues(1)-itsEmin;
    itsEmin=eigenValues(1);
    itsGapE=eigenValues(2)-eigenValues(1);
    Update(itsEigenSolver.GetEigenVector(1));

    //cout << "eigenValues=" <<  eigenValues << endl;
    //cout << "eigenVector(1)=" <<  Heff.GetColumn(1) << endl;


    //    int ierr=0;
//    ch(Heff, eigenValues ,true,ierr);
//    assert(ierr==0);
;
}

MatrixProductSite::Vector3T MatrixProductSite::IterateLeft_F(const SiteOperator* so, const Vector3T& Fam1,bool cache)
{
    TensorNetworks::ipairT Dw=so->GetDw();
    int Dw2=Dw.second;
    Vector3T F(Dw2,itsD2,itsD2,1);
    for (int w2=1;w2<=Dw2;w2++)
        for (int i2=1;i2<=itsD2;i2++)
            for (int j2=1;j2<=itsD2;j2++)
                F(w2,i2,j2)=ContractAWFA(w2,i2,j2,so,Fam1);
    if (cache) itsHLeft_Cache=F;
    return F;
}

MatrixProductSite::Vector4T MatrixProductSite::IterateLeft_F(const SiteOperator* so1, const SiteOperator* so2, const Vector4T& Fam1) const
{
    const Dw12& Dws1=so1->GetDw12();
    const Dw12& Dws2=so2->GetDw12();

    Vector4T F(Dws1.Dw2,Dws2.Dw2,itsD2,itsD2,1);
    for (int w2=1;w2<=Dws1.Dw2;w2++)
    for (int v2=1;v2<=Dws2.Dw2;v2++)
        for (int i2=1;i2<=itsD2;i2++)
            for (int j2=1;j2<=itsD2;j2++)
                F(w2,v2,i2,j2)=ContractAWWFA(w2,v2,i2,j2,so1,so2,Fam1);
    return F;
}

MatrixProductSite::eType MatrixProductSite::ContractAWFA(int w2, int i2, int j2, const SiteOperator* so, const Vector3T& Fam1) const
{
    eType awfa(0.0);
     for (int m=0; m<itsp; m++)
        for (int i1=1;i1<=itsD1;i1++)
            awfa+=conj(itsAs[m](i1,i2))*ContractWFA(m,w2,i1,j2,so,Fam1);

    return awfa;
}

MatrixProductSite::eType MatrixProductSite::
ContractAWWFA(int w2, int v2, int i2, int j2, const SiteOperator* so1, const SiteOperator* so2,const Vector4T& Fam1) const
{
    eType awwfa(0.0);
     for (int m=0; m<itsp; m++)
        for (int i1=1;i1<=itsD1;i1++)
            awwfa+=conj(itsAs[m](i1,i2))*ContractWWFA(m,w2,v2,i1,j2,so1,so2,Fam1);

    return awwfa;
}

MatrixProductSite::eType MatrixProductSite::ContractWWFA(int m, int w2, int v2, int i1, int j2, const SiteOperator* so1, const SiteOperator* so2, const Vector4T& Fam1) const
{
    const Dw12& Dws1=so1->GetDw12();
    eType wwfa(0.0);
    for (int o=0; o<itsp; o++)
    {
        const MatrixT& Wmo=so1->GetW(m,o);
        assert(Wmo.GetNumRows()==Dws1.Dw1);
        for (int w1=1;w1<Dws1.w1_first(w2);w1++)
            assert(Wmo(w1,w2)==0);
        for (int w1=Dws1.w1_first(w2);w1<=Dws1.Dw1;w1++)
            if (Wmo(w1,w2)!=0.0)
            {
                assert(fabs(Wmo(w1,w2))>0.0); //Make sure -0 doesn't slip through the gate
                wwfa+=Wmo(w1,w2)*ContractWFA(o,w1,v2,i1,j2,so2,Fam1);
            }
    }
    return wwfa;
}

MatrixProductSite::eType MatrixProductSite::ContractWFA(int o, int w1, int v2, int i1, int j2, const SiteOperator* so, const Vector4T& Fam1) const
{
    const Dw12& Dvs1=so->GetDw12();
    eType wfa(0.0);
    for (int n=0; n<itsp; n++)
    {
        const MatrixT& Won=so->GetW(o,n);
        assert(Won.GetNumRows()==Dvs1.Dw1);
        for (int v1=1;v1<Dvs1.w1_first(v2);v1++)
            assert(Won(v1,v2)==0);
        for (int v1=Dvs1.w1_first(v2);v1<=Dvs1.Dw1;v1++)
            if (Won(v1,v2)!=0.0)
            {
                assert(fabs(Won(v1,v2))>0.0); //Make sure -0 doesn't slip through the gate
                wfa+=Won(v1,v2)*ContractFA(n,w1,v1,i1,j2,Fam1);
            }
    }
    return wfa;
}

MatrixProductSite::eType MatrixProductSite::ContractFA(int n, int w1, int v1, int i1, int j2, const Vector4T& Fam1) const
{
    eType fa(0.0);
    for (int j1=1;j1<=itsD1;j1++)
        fa+=Fam1(w1,v1,i1,j1)*itsAs[n](j1,j2);
    return fa;
}



MatrixProductSite::eType MatrixProductSite::ContractWFA(int m, int w2, int i1, int j2, const SiteOperator* so, const Vector3T& Fam1) const
{
    const Dw12& Dws1=so->GetDw12();
    eType wfa(0.0);
    for (int n=0; n<itsp; n++)
    {
        const MatrixT& Wmn=so->GetW(m,n);
        assert(Wmn.GetNumRows()==Dws1.Dw1);
        for (int w1=1;w1<Dws1.w1_first(w2);w1++)
            assert(Wmn(w1,w2)==0);
        for (int w1=Dws1.w1_first(w2); w1<=Dws1.Dw1; w1++)
            if (Wmn(w1,w2)!=0.0)
            {
                assert(fabs(Wmn(w1,w2))>0.0);
                wfa+=Wmn(w1,w2)*ContractFA(n,w1,i1,j2,Fam1);
            }
    }
    return wfa;
}

MatrixProductSite::eType MatrixProductSite::ContractFA(int n, int w1, int i1, int j2, const Vector3T& Fam1) const
{
    eType fa(0.0);
    for (int j1=1;j1<=itsD1;j1++)
        fa+=Fam1(w1,i1,j1)*itsAs[n](j1,j2);
    return fa;
}

MatrixProductSite::Vector3T MatrixProductSite::IterateRightF(const SiteOperator* so, const Vector3T& Fap1, bool cache)
{
    TensorNetworks::ipairT Dw=so->GetDw();
    int Dw1=Dw.first;
    Vector3T F(Dw1,itsD1,itsD1,1);
    for (int w1=1;w1<=Dw1;w1++)
        for (int i1=1;i1<=itsD1;i1++)
            for (int j1=1;j1<=itsD1;j1++)
                F(w1,i1,j1)=ContractBWFB(w1,i1,j1,so,Fap1);
    if (cache)
    {
        itsHRightCache=F;
//        cout << "Caching right F=" << itsHRightCache << endl << endl;
    }
    return F;
}

MatrixProductSite::eType MatrixProductSite::ContractBWFB(int w1, int i1, int j1, const SiteOperator* so, const Vector3T& Fap1) const
{
    eType bwfb(0.0);
     for (int m=0; m<itsp; m++)
        for (int i2=1;i2<=itsD2;i2++)
            bwfb+=conj(itsAs[m](i1,i2))*ContractWFB(m,w1,i2,j1,so,Fap1);

    return bwfb;
}

MatrixProductSite::eType MatrixProductSite::ContractWFB(int m, int w1, int i2, int j1, const SiteOperator* so, const Vector3T& Fap1) const
{
    const Dw12& Dws=so->GetDw12();
    eType wfb(0.0);
     for (int n=0; n<itsp; n++)
     {
        const MatrixT& Wmn=so->GetW(m,n);
        assert(Wmn.GetNumCols()==Dws.Dw2);
        for (int w2=1;w2<=Dws.w2_last(w1);w2++)
            if (Wmn(w1,w2)!=0.0)
            {
                assert(fabs(Wmn(w1,w2))>0.0);
                wfb+=Wmn(w1,w2)*ContractFB(n,w2,i2,j1,Fap1);
            }
    }
    return wfb;
}

MatrixProductSite::eType MatrixProductSite::ContractFB(int n, int w2, int i2, int j1, const Vector3T& Fap1) const
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
double MatrixProductSite::ContractHeff(const MatrixCT& Heff) const
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
        for (int i1=1; i1<=itsD1; i1++)
            for (int i2=1; i2<=itsD2; i2++)
                itsAs[m](i1,i2)=As(m,i1,i2);

    itsNumUpdates++;
}

void MatrixProductSite::UpdateCache(const SiteOperator* so, const Vector3T& HLeft, const Vector3T& HRight)
{
    itsHLeft_Cache=IterateLeft_F(so,HLeft);
    itsHRightCache=IterateRightF(so,HRight);
}


void MatrixProductSite::Contract(const VectorT& s, const MatrixCT& Vdagger)
{
    int N1=s.GetHigh(); //N1=0 on the first site.
    if (N1>0 && N1<itsD1) itsD1=N1; //The contraction below will automatically reshape the As.

    for (int in=0; in<itsp; in++)
    {
        MatrixCT temp=Contract1(s,Vdagger*itsAs[in]);
        itsAs[in].SetLimits(0,0);
        itsAs[in]=temp; //Shallow copy
//        cout << "A[" << in << "]=" << itsAs[in] << endl;
        assert(itsAs[in].GetNumRows()==itsD1); //Verify shape is correct;
    }
}

void MatrixProductSite::Contract(const MatrixCT& U, const VectorT& s)
{
    int N1=s.GetHigh(); //N1=0 on the first site.
    if (N1>0 && N1<itsD2)
        itsD2=N1; //The contraction below will automatically reshape the As.
    for (int in=0; in<itsp; in++)
    {
        MatrixCT temp=Contract1(itsAs[in]*U,s);
        itsAs[in].SetLimits(0,0);
        itsAs[in]=temp; //Shallow copy
        assert(itsAs[in].GetNumCols()==itsD2); //Verify shape is correct;
    }
}


