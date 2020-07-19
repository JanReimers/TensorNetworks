#ifndef FULLSTATE_H
#define FULLSTATE_H

#include "TensorNetworks/Typedefs.H"

class FullState
{
    protected:
        typedef TensorNetworks::Matrix4T Matrix4T;

    public:
        FullState() {}
        virtual ~FullState() {}

        virtual long int GetSize() const=0;
        virtual void Contract(const Matrix4T& Hlocal)=0;
        virtual double Normalize()=0;

        virtual std::ostream& Dump(std::ostream&) const=0;
        friend std::ostream& operator<<(std::ostream& os,const FullState& psi)
        {
            return psi.Dump(os);
        }

    protected:

    private:
};

#endif // FULLSTATE_H
