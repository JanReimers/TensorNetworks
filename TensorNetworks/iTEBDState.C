#include "iTEBDState.H"
namespace TensorNetworks
{

void iTEBDState::Normalize(Direction lr)
{
    NormalizeAndCompress(lr,0);
}

}
