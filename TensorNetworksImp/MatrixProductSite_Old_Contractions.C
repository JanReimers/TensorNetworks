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

