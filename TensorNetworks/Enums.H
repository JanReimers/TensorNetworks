#ifndef ENUMS_H
#define ENUMS_H

#include "oml/imp/shape.h"

namespace TensorNetworks
{
    enum Position     {PLeft,PBulk,PRight};
    enum Direction    {DLeft,DRight};
    enum State        {Product,Random,Neel,Constant};
    enum SpinOperator {Sx,Sy,Sz,Sp,Sm};
    enum Trotter      {Odd,Even,TrotterX,TrotterY,TrotterZ};
    enum TrotterOrder {TNone,FirstOrder,SecondOrder,FourthOrder};
    enum iTEBDType    {Gates,MPOs,iMPOs};
    enum CompressType {CNone,Std,Parker};
    enum MPOForm      {FUnknown,RegularUpper,RegularLower,expH,FUnit};

    using TriType=Store;
};

#endif // ENUMS_H
