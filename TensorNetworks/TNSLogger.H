#ifndef TNSLOGGER_H
#define TNSLOGGER_H

#include "Containers/c_str.H"

#include <memory>

namespace TensorNetworks
{
//
// Abstract interface for a logger.  Hopefully file, console and GUIs loggers can all derive
// from this.
//
class TNSLogger
{
    public:
        TNSLogger() {};
        virtual ~TNSLogger() {};

        virtual void LogInfo(int level,c_str message)=0; //comment name to avoid warning
        virtual void LogInfo(int level,int site,c_str message)=0; //comment name to avoid warning
        virtual void LogWarn(int level,c_str message)=0;

        void LogInfo(int level,int site,const std::string& message) {LogInfo(level,site,message.c_str());} //comment name to avoid warning
        void LogInfo(int level,const std::string& message) {LogInfo(level,message.c_str());} //comment name to avoid warning
//        void LogWarn(int level,const std::string& message) {LogIn(level,message.c_str());} //comment name to avoid warning


        template<typename... Args>
        void LogInfoV(int level,c_str fmt, const Args &... args)
        {
            std::string fmt1=AddSpace(level,fmt);
            size_t n = snprintf( nullptr, 0, fmt1.c_str() ,args...) + 1; // Extra space for '\0'
            std::unique_ptr<char[]> buf( new char[ n ] );
            snprintf( buf.get(), n, fmt1.c_str(), args ... );
            LogInfo(level,buf.get());
        }
        template<typename... Args>
        void LogWarnV(int level,c_str fmt, const Args &... args)
        {
            std::string fmt1=AddSpace(level,fmt);
            size_t n = snprintf( nullptr, 0, fmt1.c_str() ,args...) + 1; // Extra space for '\0'
            std::unique_ptr<char[]> buf( new char[ n ] );
            snprintf( buf.get(), n,fmt1.c_str(), args ... );
            LogWarn(level,buf.get());
        }

    private:
        static std::string AddSpace(int level,c_str fmt);
};

extern TNSLogger* Logger;

}
#endif // TNSLOGGER_H
