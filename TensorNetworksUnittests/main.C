#include "Tests.H"
#include "TensorNetworks/Factory.H"
#include "TensorNetworksImp/SPDLogger.H"
#include <complex>

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    TensorNetworks::SPDLogger itsLogger(-1);
//    testing::GTEST_FLAG(filter) = "ExactDiagTests.*";
//    testing::GTEST_FLAG(filter) = "ExpectationsTests.*";
//    testing::GTEST_FLAG(filter) = "LinearAlgebraTests.SVDUpperTriangular";
//    testing::GTEST_FLAG(filter) = "MPSTests*";
//    testing::GTEST_FLAG(filter) = "MPSNormTests.*";
//    testing::GTEST_FLAG(filter) = "MPOTests.TestMPOStdCompressForH*";
//    testing::GTEST_FLAG(filter) = "OvMTests.*";
//    testing::GTEST_FLAG(filter) = "MPOTests.*:OvMTests.*";
//    testing::GTEST_FLAG(filter) = "MPOTests.*";
//    testing::GTEST_FLAG(filter) = "VariationalGroundStateTests.*";
//    testing::GTEST_FLAG(filter) = "ImaginaryTimeTests.*";
//    testing::GTEST_FLAG(filter) = "MPOTests.*:VariationalGroundStateTests.*:iTEBDTests.*:ExpectationsTests.*";
//    testing::GTEST_FLAG(filter) = "ITensorTests*";
//    testing::GTEST_FLAG(filter) = "BenchmarkTests.*";

    return RUN_ALL_TESTS();
}
