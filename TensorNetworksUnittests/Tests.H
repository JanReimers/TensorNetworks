#ifndef TESTS_H_INCLUDED
#define TESTS_H_INCLUDED

#include "gtest/gtest.h"
#include "TensorNetworks/MatrixProductState.H"


template <class Ob> std::string ToString(const Ob& result)
{
    std::stringstream res_stream;
    res_stream << result;
    return res_stream.str();
}

void VerifyUnit(const MatrixProductSite::MatrixT& Norm, double eps);



#endif // TESTS_H_INCLUDED
