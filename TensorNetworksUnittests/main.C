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
//    testing::GTEST_FLAG(filter) = "LinearAlgebraTests.LapackLinearSolver*";
//    testing::GTEST_FLAG(filter) = "MPSTests*";
//    testing::GTEST_FLAG(filter) = "MPSNormTests.*";
//    testing::GTEST_FLAG(filter) = "MPOTests.*";
//    testing::GTEST_FLAG(filter) = "iMPOTests.iCompress_*";
//    testing::GTEST_FLAG(filter) = "iMPOTests.*";
//    testing::GTEST_FLAG(filter) = "OvMTests.*";
//    testing::GTEST_FLAG(filter) = "iMPOTests.*:MPOTests.*:OvMTests.*";
//    testing::GTEST_FLAG(filter) = "MPOTests.DoBuildMPO_Neel";
//    testing::GTEST_FLAG(filter) = "VariationalGroundStateTests.*";
    testing::GTEST_FLAG(filter) = "iVUMPSTests.*";
//    testing::GTEST_FLAG(filter) = "iVUMPSTests.TestFindFerroGS_S12_D2_L1_h01";
//    testing::GTEST_FLAG(filter) = "ImaginaryTimeTests.*";
//    testing::GTEST_FLAG(filter) = "iTEBDTests.TestRandomEnergy*";
//    testing::GTEST_FLAG(filter) = "ITensorTests*";
//    testing::GTEST_FLAG(filter) = "BenchmarkTests.*";

    return RUN_ALL_TESTS();
}
