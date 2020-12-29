#include "NullLogger.H"

TensorNetworks::NullLogger defaultNullLogger(0);

namespace TensorNetworks
{
TNSLogger* Logger=nullptr;

NullLogger::NullLogger(int level)
{
    //Defulat logger is stdout
    if (!Logger) Logger=this;
}

NullLogger::~NullLogger()
{
    if (Logger==this) Logger=nullptr;
}

void NullLogger::LogInfo(int level,c_str message)
{

}

void NullLogger::LogInfo(int level,int site, c_str message)
{

}

void NullLogger::LogWarn(int level,c_str message)
{

}


} // namespace
