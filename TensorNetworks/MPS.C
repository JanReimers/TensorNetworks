#include "TensorNetworks/MPS.H"

namespace TensorNetworks
{

void MPS::Normalize(Direction lr)
{
    NormalizeAndCompress(lr,0);
}

}
