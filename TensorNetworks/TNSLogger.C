#include "TNSLogger.H"
#include <sstream>

namespace TensorNetworks
{

std::string TNSLogger::AddSpace(int level,c_str fmt)
{
    std::ostringstream os;
    for (int i=1;i<=level;i++) os << "   ";
    os << fmt;
    return os.str();
}

}
