#include "SPDLogger.H"

SPDLogger::SPDLogger(int level)
    : itsLevel(level)
{
    //Defulat logger is stdout
}

SPDLogger::~SPDLogger()
{
    //dtor
}

void SPDLogger::ReadyToStart(c_str message)
{
    spdlog::info("Start {}", message);
}

void SPDLogger::LogInfo(int level,c_str message)
{
    if (itsLevel>=level)
    {
        spdlog::info(message);
    }
}

void SPDLogger::LogInfo(int level,int site, c_str message)
{
    if (itsLevel>=level)
    {
        spdlog::info("Site {}: {}",site,message);
    }
}

/*void SPDLogger::LogInfo(int level,c_str message, int isite)
{
//    if  (isite>0)
//    {
//        if (level>3)
//            spdlog::debug("Site {}, {}",isite,message);
//        else
//            spdlog::info ("Site {}, {}",isite,message);
//    }
//    else
//    {
//        if (level>3)
//            spdlog::debug("{}",message);
//        else
//            spdlog::info ("{}",message);
//    }
}
*/
