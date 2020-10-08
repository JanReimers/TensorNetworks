#include "iTEBDState.H"

void iTEBDState::Normalize(TensorNetworks::Direction lr)
{
    NormalizeAndCompress(lr,0);
}
