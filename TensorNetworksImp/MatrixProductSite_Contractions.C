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


MatrixProductSite::MatrixCT  MatrixProductSite::Reshape(TensorNetworks::Direction lr)
{
    MatrixCT A;
    switch (lr)
    {
    case TensorNetworks::DLeft:
    {
        A.SetLimits(itsp*itsD1,itsD2);
        int i2_1=1;
        for (int in=0; in<itsp; in++)
            for (int i1=1; i1<=itsD1; i1++,i2_1++)
                for (int i2=1; i2<=itsD2; i2++)
                    A(i2_1,i2)=itsAs[in](i1,i2);
        break;
    }
    case TensorNetworks::DRight:
    {
        A.SetLimits(itsD1,itsp*itsD2);
        int i2_2=1;
        for (int in=0; in<itsp; in++)
            for (int i2=1; i2<=itsD2; i2++,i2_2++)
                for (int i1=1; i1<=itsD1; i1++)
                    A(i1,i2_2)=itsAs[in](i1,i2);
        break;
    }
    }
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

void  MatrixProductSite::Reshape(TensorNetworks::Direction lr,const MatrixCT& UV)
{
    switch (lr)
    {
    case TensorNetworks::DLeft:
    {
        //  If U has less columns than the As then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (UV.GetNumCols()<itsD2) Reshape(itsD1,UV.GetNumCols());//This throws away the old data
        int i2_1=1;
        for (int in=0; in<itsp; in++)
            for (int i1=1; i1<=itsD1; i1++,i2_1++)
                for (int i2=1; i2<=itsD2; i2++)
                    itsAs[in](i1,i2)=UV(i2_1,i2);
        break;
    }
    case TensorNetworks::DRight:
    {
        //  If Vdagger has less row than the As then we need to reshape the whole site.
        //  Typically this will happen at the edges of the lattice.
        //
        if (UV.GetNumRows()<itsD1) Reshape(UV.GetNumRows(),itsD2,false);//This throws away the old data
        int i2_2=1;
        for (int in=0; in<itsp; in++)
            for (int i2=1; i2<=itsD2; i2++,i2_2++)
                for (int i1=1; i1<=itsD1; i1++)
                    itsAs[in](i1,i2)=UV(i1,i2_2);
        break;
    }
    }
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

MatrixProductSite::MatrixCT MatrixProductSite::GetNorm(TensorNetworks::Direction lr) const
{
    MatrixCT ret;
    switch(lr)
    {
    case TensorNetworks::DLeft:
    {
        ret.SetLimits(itsD2,itsD2);
        Fill(ret,std::complex<double>(0.0));
        //
        //  Sum_ip A^t(ip) * A(ip)
        //
        for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
            ret+=conj(Transpose((*ip)))*(*ip);
        break;
    }
    case TensorNetworks::DRight:
    {
        ret.SetLimits(itsD1,itsD1);
        Fill(ret,std::complex<double>(0.0));
        //
        //  Sum_ip A(ip)*A^t(ip)
        //
        for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
            ret+=(*ip)*conj(Transpose((*ip)));
        break;
    }
    }
    return ret;
}
//
//  Sum_ip A^t(ip) * A(ip)
//
/*MatrixProductSite::MatrixCT MatrixProductSite::GetLeftNorm() const
{
    MatrixCT ret(itsD2,itsD2);
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
MatrixProductSite::MatrixCT MatrixProductSite::GetRightNorm() const
{
    MatrixCT ret(itsD1,itsD1);
    Fill(ret,std::complex<double>(0.0));
    for (cpIterT ip=itsAs.begin(); ip!=itsAs.end(); ip++)
        ret+=(*ip)*conj(Transpose((*ip)));
    return ret;
}

*/
MatrixProductSite::Matrix6T MatrixProductSite::
GetHeff(const SiteOperator* mops,const Vector3T& L,const Vector3T& R) const
{
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

MatrixProductSite::MatrixCT MatrixProductSite::IterateLeft_F(const MatrixProductSite* Psi2, const MatrixCT& Fam1,bool cache) const
{
//    cout << "IterateLeft_F D1,D2,DwD1,DwD2,Fam=" << itsAs[0].GetLimits() << " " << Psi2->itsAs[0].GetLimits() << " " << Fam1.GetLimits() << endl;
    assert(Fam1.GetNumRows()==      itsD1);
    assert(Fam1.GetNumCols()==Psi2->itsD1);
    MatrixCT F(itsD2,Psi2->itsD2);
    Fill(F,eType(0.0));
    for (int m=0; m<itsp; m++)
        for (int i2=1; i2<=itsD2; i2++)
            for (int j2=1; j2<=Psi2->itsD2; j2++)
                for (int i1=1; i1<=itsD1; i1++)
                    for (int j1=1; j1<=Psi2->itsD1; j1++)
                        F(i2,j2)+=Fam1(i1,j1)*conj(itsAs[m](i1,i2))*Psi2->itsAs[m](j1,j2); //Not Optimized
    if (cache) itsLeft_Cache=F;
//    cout << "Lcache=" << itsLeft_Cache.GetLimits() << endl;
    return F;
}

MatrixProductSite::MatrixCT MatrixProductSite::IterateRightF(const MatrixProductSite* Psi2, const MatrixCT& Fap1,bool cache) const
{
//    cout << "IterateRightF D1,D2,DwD1,DwD2,Fap=" << itsAs[0].GetLimits() << " " << Psi2->itsAs[0].GetLimits() << " " << Fap1.GetLimits() << endl;
    MatrixCT F(itsD1,Psi2->itsD1);
    Fill(F,eType(0.0));
//    cout << "F=" << F.GetLimits() << " Fap1=" << Fap1.GetLimits() << endl;
    assert(Fap1.GetNumRows()>=      itsD2);
    assert(Fap1.GetNumCols()==Psi2->itsD2);

    for (int m=0; m<itsp; m++)
        for (int i1=1; i1<=itsD1; i1++)
            for (int j1=1; j1<=Psi2->itsD1; j1++)
                for (int i2=1; i2<=itsD2; i2++)
                    for (int j2=1; j2<=Psi2->itsD2; j2++)
                        F(i1,j1)+=Fap1(i2,j2)*conj(itsAs[m](i1,i2))*Psi2->itsAs[m](j1,j2); //Not Optimized

    if (cache) itsRightCache=F;
//    cout << "Rcache=" << itsRightCache.GetLimits() << endl;

    return F;
}

MatrixProductSite::MatrixCT MatrixProductSite::IterateF(TensorNetworks::Direction lr ,const MatrixCT& Mold) const
{
//    cout << "IterateF D1,D2 Mold=" << itsD1 << "," << itsD2 << " " << Mold.GetLimits() << endl;
    MatrixCT M;
    if (lr==TensorNetworks::DLeft)
    {
        assert(Mold.GetNumRows()>=itsD1);
        assert(Mold.GetNumCols()>=itsD1);
        if (Mold.GetNumRows()!=itsD1 || Mold.GetNumCols()!=itsD1)
        {
            cout << "D1,D2,Mold=" << itsD1 << " " << itsD2 << " " << Mold.GetLimits() << endl;
        }
        M.SetLimits(itsD2,itsD2);
        Fill(M,eType(0.0));
//    cout << "F=" << F.GetLimits() << " Fap1=" << Fap1.GetLimits() << endl;
        for (int m=0; m<itsp; m++)
            for (int i2=1; i2<=itsD2; i2++)
                for (int j2=1; j2<=itsD2; j2++)
                    for (int i1=1; i1<=itsD1; i1++)
                        for (int j1=1; j1<=itsD1; j1++)
                            M(i2,j2)+=Mold(i1,j1)*conj(itsAs[m](i1,i2))*itsAs[m](j1,j2); //Not Optimized

    }
    else
    {
        assert(lr==TensorNetworks::DRight);
        assert(Mold.GetNumRows()==itsD2);
        assert(Mold.GetNumCols()==itsD2);
        M.SetLimits(itsD1,itsD1);
        Fill(M,eType(0.0));
        for (int m=0; m<itsp; m++)
            for (int i1=1; i1<=itsD1; i1++)
                for (int j1=1; j1<=itsD1; j1++)
                    for (int i2=1; i2<=itsD2; i2++)
                        for (int j2=1; j2<=itsD2; j2++)
                            M(i1,j1)+=Mold(i2,j2)*conj(itsAs[m](i1,i2))*itsAs[m](j1,j2); //Not Optimized
    }
    return M;
}

MatrixProductSite::Vector3T MatrixProductSite::IterateLeft_F(const SiteOperator* so, const Vector3T& Fam1,bool cache)
{
    int Dw2=so->GetDw12().Dw2;
    Vector3T F(Dw2,itsD2,itsD2,1);
    for (int w2=1; w2<=Dw2; w2++)
        for (int i2=1; i2<=itsD2; i2++)
            for (int j2=1; j2<=itsD2; j2++)
                F(w2,i2,j2)=ContractAWFA(w2,i2,j2,so,Fam1);
    if (cache) itsHLeft_Cache=F;
    return F;
}

MatrixProductSite::Vector4T MatrixProductSite::IterateLeft_F(const SiteOperator* so1, const SiteOperator* so2, const Vector4T& Fam1) const
{
    const Dw12& Dws1=so1->GetDw12();
    const Dw12& Dws2=so2->GetDw12();

    Vector4T F(Dws1.Dw2,Dws2.Dw2,itsD2,itsD2,1);
    for (int w2=1; w2<=Dws1.Dw2; w2++)
        for (int v2=1; v2<=Dws2.Dw2; v2++)
            for (int i2=1; i2<=itsD2; i2++)
                for (int j2=1; j2<=itsD2; j2++)
                    F(w2,v2,i2,j2)=ContractAWWFA(w2,v2,i2,j2,so1,so2,Fam1);
    return F;
}

MatrixProductSite::eType MatrixProductSite::ContractAWFA(int w2, int i2, int j2, const SiteOperator* so, const Vector3T& Fam1) const
{
    eType awfa(0.0);
    for (int m=0; m<itsp; m++)
        for (int i1=1; i1<=itsD1; i1++)
            awfa+=conj(itsAs[m](i1,i2))*ContractWFA(m,w2,i1,j2,so,Fam1);

    return awfa;
}

MatrixProductSite::eType MatrixProductSite::
ContractAWWFA(int w2, int v2, int i2, int j2, const SiteOperator* so1, const SiteOperator* so2,const Vector4T& Fam1) const
{
    eType awwfa(0.0);
    for (int m=0; m<itsp; m++)
        for (int i1=1; i1<=itsD1; i1++)
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
        for (int w1=1; w1<Dws1.w1_first(w2); w1++)
            assert(Wmo(w1,w2)==0);
        for (int w1=Dws1.w1_first(w2); w1<=Dws1.Dw1; w1++)
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
        for (int v1=1; v1<Dvs1.w1_first(v2); v1++)
            assert(Won(v1,v2)==0);
        for (int v1=Dvs1.w1_first(v2); v1<=Dvs1.Dw1; v1++)
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
    for (int j1=1; j1<=itsD1; j1++)
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
        for (int w1=1; w1<Dws1.w1_first(w2); w1++)
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
    for (int j1=1; j1<=itsD1; j1++)
        fa+=Fam1(w1,i1,j1)*itsAs[n](j1,j2);
    return fa;
}



MatrixProductSite::Vector3T MatrixProductSite::IterateRightF(const SiteOperator* so, const Vector3T& Fap1, bool cache)
{
    int Dw1=so->GetDw12().Dw1;;
    Vector3T F(Dw1,itsD1,itsD1,1);
    for (int w1=1; w1<=Dw1; w1++)
        for (int i1=1; i1<=itsD1; i1++)
            for (int j1=1; j1<=itsD1; j1++)
                F(w1,i1,j1)=ContractBWFB(w1,i1,j1,so,Fap1);
    if (cache) itsHRightCache=F;
    return F;
}

MatrixProductSite::eType MatrixProductSite::ContractBWFB(int w1, int i1, int j1, const SiteOperator* so, const Vector3T& Fap1) const
{
    eType bwfb(0.0);
    for (int m=0; m<itsp; m++)
        for (int i2=1; i2<=itsD2; i2++)
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
        for (int w2=1; w2<=Dws.w2_last(w1); w2++)
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
    for (int j2=1; j2<=itsD2; j2++)
        fb+=Fap1(w2,i2,j2)*itsAs[n](j1,j2);
    return fb;
}


//Left
/*void MatrixProductSite::Contract(const VectorT& s, const MatrixCT& Vdagger)
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

// Right
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
*/
void MatrixProductSite::Contract(TensorNetworks::Direction lr,const VectorT& s, const MatrixCT& UV)
{
    switch (lr)
    {
    case TensorNetworks::DRight:
    {
        int N1=s.GetHigh(); //N1=0 on the first site.
        if (N1>0 && N1<itsD2)
        {
            if (itsAs[0].GetNumCols()!=UV.GetNumRows())
                Reshape(itsD1,N1,true);
            else
                itsD2=N1; //The contraction below will automatically reshape the As.
        }
        for (int in=0; in<itsp; in++)
        {
            assert(itsAs[in].GetNumCols()==UV.GetNumRows());
            MatrixCT temp=Contract1(itsAs[in]*UV,s);
            itsAs[in].SetLimits(0,0);
            itsAs[in]=temp; //Shallow copy
            assert(itsAs[in].GetNumCols()==itsD2); //Verify shape is correct;
        }
        break;
    }
    case TensorNetworks::DLeft:
    {
        int N1=s.GetHigh(); //N1=0 on the first site.
        if (N1>0 && N1<itsD1)
        {
            if (itsAs[0].GetNumRows()!=UV.GetNumCols())
                Reshape(N1,itsD2,true);
            else
                itsD1=N1; //The contraction below will automatically reshape the As.
        }

        for (int in=0; in<itsp; in++)
        {
            assert(UV.GetNumCols()==itsAs[in].GetNumRows());
            MatrixCT temp=Contract1(s,UV*itsAs[in]);
            itsAs[in].SetLimits(0,0);
            itsAs[in]=temp; //Shallow copy
            //        cout << "A[" << in << "]=" << itsAs[in] << endl;
            assert(itsAs[in].GetNumRows()==itsD1); //Verify shape is correct;
        }
        break;
    }

    }
}



MatrixProductSite::MatrixCT MatrixProductSite::CalculateOneSiteDM()
{
    MatrixCT ro(itsp,itsp); //These can't be zero based if we want run them through eigen routines, which are hard ocded for 1 based matricies
    Fill(ro,eType(0.0));
    for (int m=0; m<itsp; m++)
        for (int n=0; n<itsp; n++)
            for (int j1=1; j1<=itsD1; j1++)
                for (int j2=1; j2<=itsD2; j2++)
                    ro(m+1,n+1)+=std::conj(itsAs[m](j1,j2))*itsAs[n](j1,j2);
    return ro;
}

MatrixProductSite::MatrixCT MatrixProductSite::InitializeTwoSiteDM(int m, int n)
{
    MatrixCT C(itsD2,itsD2);
    Fill(C,eType(0.0));
    for (int i2=1; i2<=itsD2; i2++)
        for (int j2=1; j2<=itsD2; j2++)
            for (int i1=1; i1<=itsD1; i1++)
                C(i2,j2)+=std::conj(itsAs[m](i1,i2))*itsAs[n](i1,j2);
    return C;
}

MatrixProductSite::MatrixCT MatrixProductSite::IterateTwoSiteDM(MatrixCT& Cmn)
{
    MatrixCT ret(itsD2,itsD2);
    Fill(ret,eType(0.0));
    for (int n2=0; n2<itsp; n2++)
    {
        MatrixCT CAmn=ContractCA(n2,Cmn);
        for (int i2=1; i2<=itsD2; i2++)
            for (int j2=1; j2<=itsD2; j2++)
                for (int i1=1; i1<=itsD1; i1++)
                    ret(i2,j2)+=std::conj(itsAs[n2](i1,i2))*CAmn(i1,j2);
    }
    return ret;
}

MatrixProductSite::MatrixCT MatrixProductSite::ContractCA(int n2, const MatrixCT& Cmn) const
{
    MatrixCT ret(itsD1,itsD2);
    Fill(ret,eType(0.0));
    for (int j2=1; j2<=itsD2; j2++)
        for (int i1=1; i1<=itsD1; i1++)
            for (int j1=1; j1<=itsD1; j1++)
                ret(i1,j2)+=Cmn(i1,j1)*itsAs[n2](j1,j2);
    return ret;
}

MatrixProductSite::MatrixCT MatrixProductSite::FinializeTwoSiteDM(const MatrixCT & Cmn)
{
    MatrixCT ret(itsp,itsp);
    Fill(ret,eType(0.0));
    for (int n2=0; n2<itsp; n2++)
    {
        MatrixCT CAmn=ContractCA(n2,Cmn);
        for (int m2=0; m2<itsp; m2++)
            for (int j2=1; j2<=itsD2; j2++)
                for (int i1=1; i1<=itsD1; i1++)
                    ret(m2+1,n2+1)+=std::conj(itsAs[m2](i1,j2))*CAmn(i1,j2);
    }


    return ret;
}

void MatrixProductSite::Contract(pVectorT& newAs,const SiteOperator* so)
{
    newAs.clear();
    const Dw12& Dws=so->GetDw12();
    int newD1=itsD1*Dws.Dw1;
    int newD2=itsD2*Dws.Dw2;

    for (int n=0; n<itsp; n++)
    {
        newAs.push_back(MatrixCT(newD1,newD2));
        Fill(newAs[n],eType(0.0));
        for (int m=0; m<itsp; m++)
        {
            const MatrixT& W=so->GetW(n,m);
            assert(W.GetNumRows()==Dws.Dw1);
            assert(W.GetNumCols()==Dws.Dw2);
            int i1=1; //i1=(w1,j1)
            for (int w1=1; w1<=Dws.Dw1; w1++)
                for (int j1=1; j1<=itsD1; j1++,i1++)
                {
                    int i2=1; //i2=(w2,j2)
                    for (int w2=1; w2<=Dws.Dw2; w2++)
                        for (int j2=1; j2<=itsD2; j2++,i2++)
                            newAs[n](i1,i2)+=W(w1,w2)*itsAs[m](j1,j2);
                }
        }
        //  cout << "newAs[" << n << "]=" << newAs[n] << endl;
    }

}

void  MatrixProductSite::ApplyInPlace(const SiteOperator* so)
{
    pVectorT newAs;
    Contract(newAs,so);

    const Dw12& Dws=so->GetDw12();
    itsD1=itsD1*Dws.Dw1;
    itsD2=itsD2*Dws.Dw2;
    itsAs=newAs;
}

void  MatrixProductSite::Apply(const SiteOperator* so, MatrixProductSite* psiPrime)
{
    Contract(psiPrime->itsAs,so);

    const Dw12& Dws=so->GetDw12();
    psiPrime->itsD1=itsD1*Dws.Dw1;
    psiPrime->itsD2=itsD2*Dws.Dw2;
}

