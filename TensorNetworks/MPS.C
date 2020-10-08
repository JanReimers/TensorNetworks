#include "TensorNetworks/MPS.H"

void MPS::Normalize(TensorNetworks::Direction lr)
{
    NormalizeAndCompress(lr,0);
}
