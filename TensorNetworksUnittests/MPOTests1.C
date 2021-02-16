#include "Tests.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworksImp/Hamiltonians/Hamiltonian_1D_NN_Heisenberg.H"
#include "Operators/OperatorElement.H"
#include "Operators/OperatorValuedMatrix.H"


//#include "TensorNetworksImp/MPS/MPSImp.H"
//#include "TensorNetworks/iTEBDState.H"
//#include "Operators/MPO_SpatialTrotter.H"
//#include "Containers/Matrix4.H"
//#include "TensorNetworks/Hamiltonian.H"
//#include "TensorNetworks/iHamiltonian.H"
//#include "TensorNetworks/SiteOperator.H"
//#include "TensorNetworks/MPO.H"
//#include "TensorNetworks/iMPO.H"
//#include "Operators/SiteOperatorImp.H"
//#include "Containers/Matrix6.H"

using dcmplx=TensorNetworks::dcmplx;
using TensorNetworks::MatrixOR;
using TensorNetworks::TriType;
using TensorNetworks::Lower;
using TensorNetworks::Upper;
using TensorNetworks::Direction;
using TensorNetworks::DLeft;
using TensorNetworks::DRight;
using TensorNetworks::SVCompressorR;
using TensorNetworks::MatrixRT;
using TensorNetworks::Position;
using TensorNetworks::PBulk;
using TensorNetworks::PLeft;
using TensorNetworks::PRight;

class MPOTests1 : public ::testing::Test
{
public:
//    typedef TensorNetworks::Matrix6CT Matrix6CT;
//    typedef TensorNetworks::MatrixRT  MatrixRT;
//    typedef TensorNetworks::MatrixCT  MatrixCT;
//    typedef TensorNetworks::Vector3CT Vector3CT;
//    typedef TensorNetworks::dcmplx     dcmplx;
    MPOTests1()
        : eps(2.0e-15)
        , itsFactory(TensorNetworks::Factory::GetFactory())
        , itsOperatorClient(0)
    {
        assert(itsFactory);
        StreamableObject::SetToPretty();
    }
    ~MPOTests1()
    {
        delete itsFactory;
        if (itsOperatorClient) delete itsOperatorClient;
    }

    void Setup(double S)
    {
        if (itsOperatorClient) delete itsOperatorClient;
        itsOperatorClient=new TensorNetworks::Hamiltonian_1D_NN_Heisenberg(S,1.0,1.0,0.0);
        assert(itsOperatorClient);
    }

    void TestQR (MatrixOR OvM,Direction,TriType,Position lbr);
    void TestSVD(double S,Direction,TriType);
    Direction Invert(Direction lr) const
    {
        if (lr==DLeft)
            lr=DRight;
        else if (lr==DRight)
            lr=DLeft;
        return lr;
    }

    double eps;
           TensorNetworks::Factory*         itsFactory;
    const  TensorNetworks::OperatorClient1* itsOperatorClient;
};

void MPOTests1::TestQR(MatrixOR OvM,Direction lr,TriType ul,Position lbr)
{
    auto [X1,X2]=OvM.GetChi12();
    int d=2*itsOperatorClient->GetS()+1;
    if (lbr==PLeft)
    {
        MatrixRT l(0,0,0,X2+1);
        Fill(l,0.0);
        switch (ul)
        {
        case Upper:
            l(0,0)=1.0;
            break;
        case Lower:
            l(0,X2+1)=1.0;
            break;
        }
//        cout << "OvM=" << OvM << endl;
//        cout << "l=" << l << endl;
        OvM=MatrixOR(l*OvM);
        OvM.SetUpperLower(ul);
//        cout << "OvM=" << OvM << endl;
    }
    if (lbr==PRight)
    {
        MatrixRT r(0,X1+1,0,0);
        Fill(r,0.0);
        switch (ul)
        {
        case Upper:
            r(X1+1,0)=1.0;
            break;
        case Lower:
            r(0,0)=1.0;
            break;
        }
//        cout << "OvM=" << OvM << endl;
//        cout << "r=" << r << endl;
        OvM=MatrixOR(OvM*r);
        OvM.SetUpperLower(ul);
//        cout << "OvM=" << OvM << endl;
    }
    MatrixOR V=OvM.GetV(lr);
//    cout << "V=" << V.GetLimits() << endl;
    auto [Q,R]=OvM.BlockQX(lr);
    MatrixRT R1=R;
//    cout << "Q=" << Q.GetLimits() << endl;
//    cout << "R1=" << R1.GetLimits() << endl;
    if (lr==DLeft)
        R1.SetLimits(MatLimits(Q.GetColLimits(),V.GetColLimits()),true); //Shrink R back to Q size so we can multiply.
    else if (lr==DRight)
        R1.SetLimits(MatLimits(V.GetRowLimits(),Q.GetRowLimits()),true); //Shrink R back to Q size so we can multiply.
    else
        assert(false);
//    cout << "R1=" << R1.GetLimits() << endl;
    MatrixOR V1;
    if (lr==DLeft)
        V1=Q*R1;
    else
        V1=R1*Q;
//    cout << "V=" << V << endl;
//    cout << "V1=" << V1 << endl;
    EXPECT_NEAR(MaxDelta(V,V1),0.0,d*eps);

//    cout << "Q=" << Q << endl;
    if (ul==Upper)
    {
        EXPECT_TRUE(IsUpperTriangular(R,1e-13));
        if (Q.GetNumCols()>1)
        {
            EXPECT_TRUE(IsUpperTriangular(Q));
        }
    }
    if (ul==Lower)
    {
        EXPECT_TRUE(IsLowerTriangular(R));
        if (Q.GetNumRows()>1)
        {
            EXPECT_TRUE(IsLowerTriangular(Q));
        }
    }
    EXPECT_TRUE(IsUnit(Q.GetOrthoMatrix(lr),d*eps));
    EXPECT_FALSE(IsUnit(Q.GetOrthoMatrix(Invert(lr)),d*eps));
}

void MPOTests1::TestSVD(double S,Direction lr,TriType ul)
{
    Setup(S);
    SVCompressorR* comp=itsFactory->MakeMPOCompressor(0,1e-14);

    MatrixOR OvM(itsOperatorClient->GetMatrixO(ul));
    MatrixOR V=OvM.GetV(lr);
    auto [Q,R]=OvM.BlockSVD(lr,comp);
    R.SetLimits(Q.GetLimits(),true); //Shrink R back to Q size so we can multiply.
    MatrixOR V1;
    if (lr==DLeft) V1=Q*R; else V1=R*Q;

    EXPECT_NEAR(MaxDelta(V,V1),0.0,eps);

    if (ul==Upper)
    {
//        EXPECT_TRUE(IsUpperTriangular(R)); Not guaranteed for SVD
        EXPECT_TRUE(IsUpperTriangular(Q));
    }
    if (ul==Lower)
    {
//        EXPECT_TRUE(IsLowerTriangular(R)); Not guaranteed for SVD
        EXPECT_TRUE(IsLowerTriangular(Q));
    }
    EXPECT_TRUE(IsUnit(Q.GetOrthoMatrix(lr),eps));
    EXPECT_FALSE(IsUnit(Q.GetOrthoMatrix(Invert(lr)),eps));
}

TEST_F(MPOTests1,MakeHamiltonian)
{
    Setup(0.5);
}

TEST_F(MPOTests1,OperatorElement1)
{
    double S=0.5;
    {
        TensorNetworks::OperatorSz Sz12(S);
        EXPECT_EQ(Sz12(0,0),-0.5);
        EXPECT_EQ(Sz12(1,0), 0.0);
        EXPECT_EQ(Sz12(0,1), 0.0);
        EXPECT_EQ(Sz12(1,1), 0.5);
    }

    {
        TensorNetworks::OperatorSp Sp12(S);
        EXPECT_EQ(Sp12(0,0), 0.0);
        EXPECT_EQ(Sp12(1,0), 1.0);
        EXPECT_EQ(Sp12(0,1), 0.0);
        EXPECT_EQ(Sp12(1,1), 0.0);
    }

    {
        TensorNetworks::OperatorSm Sm12(S);
        EXPECT_EQ(Sm12(0,0), 0.0);
        EXPECT_EQ(Sm12(1,0), 0.0);
        EXPECT_EQ(Sm12(0,1), 1.0);
        EXPECT_EQ(Sm12(1,1), 0.0);
    }

    {
        TensorNetworks::OperatorSy Sy12(S);
        EXPECT_EQ(Sy12(0,0), 0.0);
        EXPECT_EQ(Sy12(1,0), dcmplx(0.0,-0.5));
        EXPECT_EQ(Sy12(0,1), dcmplx(0.0, 0.5));
        EXPECT_EQ(Sy12(1,1), 0.0);
    }
    {
        TensorNetworks::OperatorSx Sx12(S);
        EXPECT_EQ(Sx12(0,0), 0.0);
        EXPECT_EQ(Sx12(1,0), 0.5);
        EXPECT_EQ(Sx12(0,1), 0.5);
        EXPECT_EQ(Sx12(1,1), 0.0);
    }
}

TEST_F(MPOTests1,OperatorValuedMatrix1)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    EXPECT_EQ(OvM(0,0),TensorNetworks::OperatorI (S));
    EXPECT_EQ(OvM(1,0),TensorNetworks::OperatorSp(S));
    EXPECT_EQ(OvM(2,0),TensorNetworks::OperatorSm(S));
    EXPECT_EQ(OvM(3,0),TensorNetworks::OperatorSz(S));
    EXPECT_EQ(OvM(0,1),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(0,2),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(0,3),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM.GetUpperLower(),Lower);
//    cout << "OvM=" << OvM << endl;
}

TEST_F(MPOTests1,OperatorValuedMatrix2)
{
    double S=1.0;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    EXPECT_EQ(OvM(0,0),TensorNetworks::OperatorI (S));
    EXPECT_EQ(OvM(1,0),TensorNetworks::OperatorSp(S));
    EXPECT_EQ(OvM(2,0),TensorNetworks::OperatorSm(S));
    EXPECT_EQ(OvM(3,0),TensorNetworks::OperatorSz(S));
    EXPECT_EQ(OvM(0,1),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(0,2),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(0,3),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM.GetUpperLower(),Lower);
}

TEST_F(MPOTests1,OperatorValuedMatrix3)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    EXPECT_EQ(OvM(0,0),TensorNetworks::OperatorI (S));
    EXPECT_EQ(OvM(0,1),TensorNetworks::OperatorSp(S));
    EXPECT_EQ(OvM(0,2),TensorNetworks::OperatorSm(S));
    EXPECT_EQ(OvM(0,3),TensorNetworks::OperatorSz(S));
    EXPECT_EQ(OvM(1,0),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(2,0),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(3,0),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM.GetUpperLower(),Upper);

//    cout << "OvM=" << OvM << endl;
}

TEST_F(MPOTests1,OperatorValuedMatrix4)
{
    double S=1.0;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    EXPECT_EQ(OvM(0,0),TensorNetworks::OperatorI (S));
    EXPECT_EQ(OvM(0,1),TensorNetworks::OperatorSp(S));
    EXPECT_EQ(OvM(0,2),TensorNetworks::OperatorSm(S));
    EXPECT_EQ(OvM(0,3),TensorNetworks::OperatorSz(S));
    EXPECT_EQ(OvM(1,0),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(2,0),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM(3,0),TensorNetworks::OperatorZ (S));
    EXPECT_EQ(OvM.GetUpperLower(),Upper);
}


TEST_F(MPOTests1,OperatorValuedMatrixGetVUpper)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    MatrixOR Vl=OvM.GetV(DLeft);
    MatrixOR Vr=OvM.GetV(DRight);
    EXPECT_EQ(OvM.GetNumRows(),Vl.GetNumRows()+1);
    EXPECT_EQ(OvM.GetNumCols(),Vl.GetNumCols()+1);
    EXPECT_EQ(OvM.GetNumRows(),Vr.GetNumRows()+1);
    EXPECT_EQ(OvM.GetNumCols(),Vr.GetNumCols()+1);
    for (index_t i:Vl.rows())
    for (index_t j:Vl.cols())
        EXPECT_EQ(OvM(i,j),Vl(i,j));
    for (index_t i:Vr.rows())
    for (index_t j:Vr.cols())
        EXPECT_EQ(OvM(i,j),Vr(i,j));
}

TEST_F(MPOTests1,OperatorValuedMatrixGetVLower)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    MatrixOR Vl=OvM.GetV(DLeft);
    MatrixOR Vr=OvM.GetV(DRight);
    EXPECT_EQ(OvM.GetNumRows(),Vl.GetNumRows()+1);
    EXPECT_EQ(OvM.GetNumCols(),Vl.GetNumCols()+1);
    EXPECT_EQ(OvM.GetNumRows(),Vr.GetNumRows()+1);
    EXPECT_EQ(OvM.GetNumCols(),Vr.GetNumCols()+1);
    for (index_t i:Vl.rows())
    for (index_t j:Vl.cols())
        EXPECT_EQ(OvM(i,j),Vl(i,j));
    for (index_t i:Vr.rows())
    for (index_t j:Vr.cols())
        EXPECT_EQ(OvM(i,j),Vr(i,j));
}

TEST_F(MPOTests1,OperatorValuedMatrixSetVUpper)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    MatrixOR Copy(OvM);
    MatrixOR Vl=OvM.GetV(DLeft);
    Copy.SetV(DLeft,Vl);
    EXPECT_EQ(OvM,Copy);
    MatrixOR Vr=OvM.GetV(DRight);
    Copy.SetV(DRight,Vr);
    EXPECT_EQ(OvM,Copy);
}

TEST_F(MPOTests1,OperatorValuedMatrixSetVLower)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    MatrixOR Copy(OvM);
    MatrixOR Vl=OvM.GetV(DLeft);
    Copy.SetV(DLeft,Vl);
    EXPECT_EQ(OvM,Copy);
    MatrixOR Vr=OvM.GetV(DRight);
    Copy.SetV(DRight,Vr);
    EXPECT_EQ(OvM,Copy);
}

TEST_F(MPOTests1,OperatorValuedMatrixFlattenUpper)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    MatrixOR Copy(OvM);
    Matrix<double> F=Copy.Flatten(DLeft);
    Copy.UnFlatten(F);
    EXPECT_EQ(OvM,Copy);
    F=Copy.Flatten(DRight);
    Copy.UnFlatten(F);
    EXPECT_EQ(OvM,Copy);
}

TEST_F(MPOTests1,OperatorValuedMatrixFlattenLower)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    MatrixOR Copy(OvM);
    Matrix<double> F=Copy.Flatten(DLeft);
    Copy.UnFlatten(F);
    EXPECT_EQ(OvM,Copy);
    F=Copy.Flatten(DRight);
    Copy.UnFlatten(F);
    EXPECT_EQ(OvM,Copy);
}

TEST_F(MPOTests1,OperatorValuedMatrixFlattenVUpper)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
    {
        MatrixOR V=OvM.GetV(DLeft);
        MatrixOR Copy(V);
        Matrix<double> F=V.Flatten(DLeft);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
        F=Copy.Flatten(DRight);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
    }
    {
        MatrixOR V=OvM.GetV(DRight);
        MatrixOR Copy(V);
        Matrix<double> F=V.Flatten(DRight);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
        F=Copy.Flatten(DRight);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
    }
}

TEST_F(MPOTests1,OperatorValuedMatrixFlattenVLower)
{
    double S=0.5;
    Setup(S);
    MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
    {
        MatrixOR V=OvM.GetV(DLeft);
        MatrixOR Copy(V);
        Matrix<double> F=V.Flatten(DLeft);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
        F=Copy.Flatten(DRight);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
    }
    {
        MatrixOR V=OvM.GetV(DRight);
        MatrixOR Copy(V);
        Matrix<double> F=V.Flatten(DRight);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
        F=Copy.Flatten(DRight);
        Copy.UnFlatten(F);
        EXPECT_EQ(V,Copy);
    }
}

TEST_F(MPOTests1,Grow1)
{
    MatrixRT A(1,4,1,24);
    Unit(A);
    MatLimits l(0,4,0,24);
    TensorNetworks::Grow(A,l);
    EXPECT_TRUE(IsUnit(A));
}
TEST_F(MPOTests1,Grow2)
{
    MatrixRT A(1,24,1,4);
    Unit(A);
    MatLimits l(0,24,0,4);
    TensorNetworks::Grow(A,l);
    EXPECT_TRUE(IsUnit(A));
}
TEST_F(MPOTests1,Grow3)
{
    MatrixRT A(0,4,0,24);
    Unit(A);
    MatLimits l(0,5,0,25);
    TensorNetworks::Grow(A,l);
    for (index_t i:A.rows())
    {
        if (i==5)
            EXPECT_EQ(A(i,25),1.0);
        else
            EXPECT_EQ(A(i,25),0.0);
    }
    for (index_t i:A.cols())
    {
        if (i==25)
            EXPECT_EQ(A(5,i),1.0);
        else
            EXPECT_EQ(A(5,i),0.0);
    }
    EXPECT_FALSE(IsUnit(A));
}
TEST_F(MPOTests1,Grow4)
{
    MatrixRT A(0,24,0,4);
    Unit(A);
    MatLimits l(0,25,0,5);
    TensorNetworks::Grow(A,l);
    for (index_t i:A.rows())
    {
        if (i==25)
            EXPECT_EQ(A(i,5),1.0);
        else
            EXPECT_EQ(A(i,5),0.0);
    }
    for (index_t i:A.cols())
    {
        if (i==5)
            EXPECT_EQ(A(25,i),1.0);
        else
            EXPECT_EQ(A(25,i),0.0);
    }
    EXPECT_FALSE(IsUnit(A));
}

TEST_F(MPOTests1,Grow5)
{
    MatrixRT A(0,4,0,4);
    Unit(A);
    MatLimits l(0,5,0,5);
    TensorNetworks::Grow(A,l);
    EXPECT_TRUE(IsUnit(A));
}

TEST_F(MPOTests1,Lower1)
{
    int M=4,N=24;
    MatrixRT A(M,N);
    Fill(A,1.0);
    EXPECT_FALSE(IsLowerTriangular(A));
    int delta=Max(1,N-M+1);
    for (index_t i:A.rows())
    for (index_t j:A.cols(i+delta))
        A(i,j)=0;

    EXPECT_TRUE(IsLowerTriangular(A));
    EXPECT_TRUE(IsLowerTriangular(A,0.0));
    EXPECT_FALSE(IsUpperTriangular(A));
    EXPECT_FALSE(IsUpperTriangular(A,0.0));

}
TEST_F(MPOTests1,Lower2)
{
    int M=24,N=4;
    MatrixRT A(M,N);
    Fill(A,1.0);
    EXPECT_FALSE(IsLowerTriangular(A));
    int delta=Max(1,N-M+1);
    for (index_t i:A.rows())
    for (index_t j:A.cols(i+delta))
        A(i,j)=0;

    EXPECT_TRUE(IsLowerTriangular(A));
    EXPECT_TRUE(IsLowerTriangular(A,0.0));
    EXPECT_FALSE(IsUpperTriangular(A));
    EXPECT_FALSE(IsUpperTriangular(A,0.0));

}

TEST_F(MPOTests1,Upper1)
{
    int M=4,N=24;
    MatrixRT A(M,N);
    Fill(A,1.0);
    EXPECT_FALSE(IsLowerTriangular(A));
    int delta=Max(1,M-N+1);
    for (index_t j:A.cols())
    for (index_t i:A.rows(j+delta))
        A(i,j)=0;

    EXPECT_TRUE(IsUpperTriangular(A));
    EXPECT_TRUE(IsUpperTriangular(A,0.0));
    EXPECT_FALSE(IsLowerTriangular(A));
    EXPECT_FALSE(IsLowerTriangular(A,0.0));

}

TEST_F(MPOTests1,Upper2)
{
    int M=24,N=4;
    MatrixRT A(M,N);
    Fill(A,1.0);
    EXPECT_FALSE(IsLowerTriangular(A));
    int delta=Max(1,M-N+1);
    for (index_t j:A.cols())
    for (index_t i:A.rows(j+delta))
        A(i,j)=0;

    EXPECT_TRUE(IsUpperTriangular(A));
    EXPECT_TRUE(IsUpperTriangular(A,0.0));
    EXPECT_FALSE(IsLowerTriangular(A));
    EXPECT_FALSE(IsLowerTriangular(A,0.0));

}

TEST_F(MPOTests1,OperatorValuedMatrixQRBulk)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
            TestQR(OvM,DLeft ,Upper,PBulk);
            TestQR(OvM,DRight,Upper,PBulk);
        }
        {
            MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
            TestQR(OvM,DLeft ,Lower,PBulk);
            TestQR(OvM,DRight,Lower,PBulk);
        }
    }
}

TEST_F(MPOTests1,OperatorValuedMatrixQRBulkH2)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Upper));
            MatrixOR H2=TensorProduct(H,H);
            TestQR(H2,DLeft ,Upper,PBulk);
            TestQR(H2,DRight,Upper,PBulk);
        }
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Lower));
            MatrixOR H2=TensorProduct(H,H);
            TestQR(H2,DLeft ,Lower,PBulk);
            TestQR(H2,DRight,Lower,PBulk);
        }
    }
}

TEST_F(MPOTests1,OperatorValuedMatrixQRLeft)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
            TestQR(OvM,DLeft ,Upper,PLeft);
        }
        {
            MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
            TestQR(OvM,DLeft ,Lower,PLeft);
        }
    }
}

TEST_F(MPOTests1,OperatorValuedMatrixQRLeftH2)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Upper));
            MatrixOR H2=TensorProduct(H,H);
            TestQR(H2,DLeft ,Upper,PLeft);
        }
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Lower));
            MatrixOR H2=TensorProduct(H,H);
            TestQR(H2,DLeft ,Lower,PLeft);
        }
    }
}

TEST_F(MPOTests1,OperatorValuedMatrixQRRight)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR OvM(itsOperatorClient->GetMatrixO(Upper));
            TestQR(OvM,DRight ,Upper,PRight);
        }
        {
            MatrixOR OvM(itsOperatorClient->GetMatrixO(Lower));
            TestQR(OvM,DRight ,Lower,PRight);
        }
    }

}

TEST_F(MPOTests1,OperatorValuedMatrixQRRightH2)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        Setup(S);
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Upper));
            MatrixOR H2=TensorProduct(H,H);
            TestQR(H2,DRight,Upper,PRight);
        }
        {
            MatrixOR H(itsOperatorClient->GetMatrixO(Lower));
            MatrixOR H2=TensorProduct(H,H);
            TestQR(H2,DRight,Lower,PRight);
        }
    }
}


TEST_F(MPOTests1,OperatorValuedMatrixSVD)
{
    for (double S=0.5;S<=2.5;S+=0.5)
    {
        TestSVD(S,DLeft ,Upper);
        TestSVD(S,DRight,Upper);
        TestSVD(S,DLeft ,Lower);
        TestSVD(S,DRight,Lower);
    }
}
