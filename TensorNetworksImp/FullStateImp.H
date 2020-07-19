#ifndef FULLSTATEIMP_H
#define FULLSTATEIMP_H

#include "TensorNetworks/FullState.H"
#include "TensorNetworksImp/Typedefs.H"

//---------------------------------------------------------------------
//
//  Store a full state vector with p^L amplitues
//
class FullStateImp : public virtual FullState
{
    public:
        FullStateImp(int L, double S);
        virtual ~FullStateImp();

        virtual long int GetSize() const {return itsN;}
        virtual void Contract(const Matrix4T& Hlocal);
        virtual double Normalize();
        virtual std::ostream& Dump(std::ostream&) const;

    protected:
        int  GetIndex(const Vector<int>& state) const;
        void ContractLocal(int isite,int ia,Vector<int>& stateVector,int& indexn,int& indexm, const Matrix4T& Hlocal);


    private:
        int      itsL;
        double   itsS;
        int      itsp;
        long int itsN; //p^L
        TensorNetworks::ArrayCT itsAmplitudes;
        TensorNetworks::ArrayCT itsAmplitudes1;


};

#endif // FULLSTATEIMP_H
