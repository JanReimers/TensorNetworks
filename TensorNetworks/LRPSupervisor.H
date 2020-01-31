#ifndef LRPSUPERVISOR_H
#define LRPSUPERVISOR_H

//
//  Supervisor for a long running process (LRP).  The public interface is for use by
//  the LRP, it simply calls ReadyToStart at the begining and then
//  repeatedly calls DoneOneStep(level) where level defines a hierarcy of steps.
//  These calls may be blocking ... that's the point.  The GUI user can step through
//  a calculation pla/pause it.  The superviser can also update the GUI at each step if required.
//
class LRPSupervisor
{
    public:
        LRPSupervisor() {};
        virtual ~LRPSupervisor() {};

        virtual void ReadyToStart() {};
        virtual void DoneOneStep(int /*level*/,int isite=-1) {(void)isite;} //comment name to avoid warning

//    protected:
//        virtual void Start() {};
//        virtual void Pause() {};
//        virtual void Step() {};
//       virtual void Stop() {};
};

#endif // LRPSUPERVISOR_H
