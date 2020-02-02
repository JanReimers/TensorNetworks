#include "oml/vector.h"

//
//  W matrices are lower triangular and in addition not dense below the diagonal
//  If contractions know Dw2(w1) then we can skip most of the zero elements.
//  We use an oml Vector to get 1 baed indexing which is the convention in the contraction code.
//
struct Dw12
{
    Dw12(int _Dw1, const Vector<int>& _Dw2)
        : Dw1(_Dw1)
        , Dw2(_Dw2)
        {}

    int Dw1;
    Vector<int> Dw2; //Dw2 as a function of w1
};
