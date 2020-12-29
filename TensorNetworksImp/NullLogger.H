#ifndef NULLLOGGER_H
#define NULLLOGGER_H

#include "TensorNetworks/TNSLogger.H"

namespace TensorNetworks
{

class NullLogger : public virtual TNSLogger
{
    public:
        NullLogger(int level);
        virtual ~NullLogger();
        virtual void LogInfo(int level,c_str message);
        virtual void LogInfo(int level,int site,c_str message);
        virtual void LogWarn(int level,c_str message);

    protected:

};

}
#endif // NULLLOGGER_H
