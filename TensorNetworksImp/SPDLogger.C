#include "TensorNetworksImp/SPDLogger.H"
#include "TensorNetworksImp/NullLogger.H"
#include "spdlog/spdlog.h"
#include <iostream>

namespace TensorNetworks
{

SPDLogger::SPDLogger(int level)
    : itsLevel(level)
{
    //Logger is null default the override.
    if (!Logger  || dynamic_cast<NullLogger*>(Logger))
    {
        Logger=this;
        std::cout << "Setting logger to SPD" << std::endl;
    }
}

SPDLogger::~SPDLogger()
{
    if (Logger==this) Logger=nullptr;
}

void SPDLogger::LogInfo(int level,c_str message)
{
    if (itsLevel>=level)
    {
        std::string m1=AddSpace(level,message);
        spdlog::info(m1);
    }
}

void SPDLogger::LogInfo(int level,int site, c_str message)
{
    if (itsLevel>=level)
    {
        std::string m1=AddSpace(level,message);
        spdlog::info("Site {}: {}",site,m1);
    }
}

void SPDLogger::LogWarn(int level,c_str message)
{
    if (itsLevel>=level)
    {
        std::string m1=AddSpace(level,message);
        spdlog::warn(m1);
    }
}


} // namespace
