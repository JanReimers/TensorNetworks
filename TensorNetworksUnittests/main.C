#include "Tests.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworksImp/SPDLogger.H"
#include <complex>

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    TensorNetworks::SPDLogger itsLogger(2);
//    testing::GTEST_FLAG(filter) = "ExactDiagTests.*";
//    testing::GTEST_FLAG(filter) = "ExpectationsTests.*";
//    testing::GTEST_FLAG(filter) = "LinearAlgebraTests.*";
//    testing::GTEST_FLAG(filter) = "MPSTests*";
//    testing::GTEST_FLAG(filter) = "MPSNormTests*";
//    testing::GTEST_FLAG(filter) = "MPOTests.*";
//    testing::GTEST_FLAG(filter) = "VariationalGroundStateTests.TestSweepL2S1D2";
//    testing::GTEST_FLAG(filter) = "ImaginaryTimeTests.TestITimeFourthOrderTrotterL2";
    testing::GTEST_FLAG(filter) = "iTEBDTests.FindiTimeGSD4S12";
//    testing::GTEST_FLAG(filter) = "ITensorTests*";
//    testing::GTEST_FLAG(filter) = "BenchmarkTests.*";

    return RUN_ALL_TESTS();
}

void VerifyUnit(const TensorNetworks::MatrixCT& Norm, double eps)
{
        int D=Norm.GetNumRows();
 //       cout << "site " << i << " has D=" << D << endl;
        TensorNetworks::MatrixCT I(D,D);
        Unit(I);
        EXPECT_NEAR(Max(fabs(real(Norm-I))),0.0,eps);
        EXPECT_NEAR(Max(fabs(imag(Norm  ))),0.0,eps);
        std::string lim="(1:";
        lim= lim + std::to_string(D) + "),(1:" + std::to_string(D) + ") ";
        EXPECT_EQ(ToString(Norm.GetLimits()),lim.c_str());
}



