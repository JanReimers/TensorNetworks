#ifndef TESTS_H_INCLUDED
#define TESTS_H_INCLUDED

#include "gtest/gtest.h"
#include "TensorNetworksImp/MatrixProductStateImp.H"
#include <iostream>
#include <string>

using std::cout;
using std::endl;


template <class Ob> std::string ToString(const Ob& result)
{
    std::stringstream res_stream;
    res_stream << result;
    return res_stream.str();
}

void VerifyUnit(const TensorNetworks::MatrixCT& Norm, double eps);



#endif // TESTS_H_INCLUDED
