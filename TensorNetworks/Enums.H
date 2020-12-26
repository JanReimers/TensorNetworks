#ifndef ENUMS_H
#define ENUMS_H

namespace TensorNetworks
{
    enum Position    {PLeft,PBulk,PRight};
    enum Direction   {DLeft,DRight};
    enum State       {Product,Random,Neel};
    enum SpinOperator{Sx,Sy,Sz,Sp,Sm};
    enum Trotter     {Odd,Even,TrotterX,TrotterY,TrotterZ};
    enum TrotterOrder{None,FirstOrder,SecondOrder,FourthOrder};
    enum iTEBDType   {Gates,MPOs,iMPOs};
};

#endif // ENUMS_H
