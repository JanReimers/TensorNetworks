#include "gtest/gtest.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
//    ::testing::GTEST_FLAG(filter) = "MatrixProductTesting*";
//    ::testing::GTEST_FLAG(filter) = "MPOTesting*";
    ::testing::GTEST_FLAG(filter) = "SVDTesting*";
    return RUN_ALL_TESTS();
}

