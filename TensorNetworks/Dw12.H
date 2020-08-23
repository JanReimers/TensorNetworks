#ifndef DW12_H
#define DW12_H
#include "oml/vector.h"

//
//  W matrices are lower triangular and in addition not dense below the diagonal
//  If contractions know Dw2(w1) and Dw1(w2) then we can skip most of the zero elements.
//  We use an oml Vector to get 1 baed indexing which is the convention in the contraction code.
//
struct Dw12
{
    Dw12();
    Dw12(int _Dw1, int _Dw2);
    Dw12(int _Dw1, int _Dw2, const Vector<int>& _w1_first, const Vector<int>& _w2_last);

    int Dw1,Dw2;
    Vector<int> w1_first; //strt index for w1 as a function of w2
    Vector<int> w2_last; //Dw2 as a function of w1
};

#endif //DW12_H
