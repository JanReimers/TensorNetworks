#include "Tests.H"
#include <complex>

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
//    testing::GTEST_FLAG(filter) = "SVDTesting*";
//    testing::GTEST_FLAG(filter) = "MatrixProductTesting*";
//    testing::GTEST_FLAG(filter) = "MPSNormTesting*";
//    testing::GTEST_FLAG(filter) = "MPOTesting*";
//    testing::GTEST_FLAG(filter) = "GroundStateTesting*";

    return RUN_ALL_TESTS();
}

void VerifyUnit(const MatrixProductSite::MatrixT& Norm, double eps)
{
        int D=Norm.GetNumRows();
 //       cout << "site " << i << " has D=" << D << endl;
        MatrixProductSite::MatrixT I(D,D);
        Unit(I);
        EXPECT_NEAR(Max(abs(real(Norm-I))),0.0,eps);
        EXPECT_NEAR(Max(abs(imag(Norm  ))),0.0,eps);
        std::string lim="(1:";
        lim= lim + std::to_string(D) + "),(1:" + std::to_string(D) + ") ";
        EXPECT_EQ(ToString(Norm.GetLimits()),lim.c_str());
}



