#include "TensorNetworks/Dw12.H"
//#include "oml/vector_io.h"
//#include <iostream>
 Dw12::Dw12()
    : Dw1(0)
    , Dw2(0)
    , w1_first(0)
    , w2_last (0)
    {}

 Dw12::Dw12(int _Dw1, int _Dw2)
    : Dw1(_Dw1)
    , Dw2(_Dw2)
    , w1_first(Dw2)
    , w2_last (Dw1)
{
    assert(w1_first.size()==Dw2);
    assert(w2_last.size()==Dw1);
    Fill(w1_first,1);
    Fill(w2_last ,Dw2);
    //std::cout << "w1_first=" << w1_first << std::endl;
    //std::cout << "w2_last=" << w2_last << std::endl;
    //assert(Min(w1_first)==1);
    //assert(Max(w2_last )==Dw2);
}

 Dw12::Dw12(int _Dw1, int _Dw2, const Vector<int>& _w1_first, const Vector<int>& _w2_last)
    : Dw1(_Dw1)
    , Dw2(_Dw2)
    , w1_first(_w1_first)
    , w2_last (_w2_last )
{
    assert(w1_first.size()==Dw2);
    assert(w2_last.size()==Dw1);
    //std::cout << "w1_first=" << w1_first << std::endl;
    //std::cout << "w2_last=" << w2_last << std::endl;
    //assert(Min(w1_first)==1);
    //assert(Max(w2_last )==Dw2);
}

