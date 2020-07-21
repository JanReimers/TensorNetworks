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
    typedef TensorNetworks::ArrayCT ArrayCT;
    public:
        FullStateImp(int L, double S);
        virtual ~FullStateImp();

        virtual long int GetSize     () const {return itsN;}
        virtual void     Normalize   (ArrayCT& amplitudes);
        virtual double   PowerIterate(const Epsilons&,const Matrix4T& Hlocal,bool quite=true);
        virtual double   Contract    (const Matrix4T& Hlocal);
        virtual double   GetE        () const {return itsE;}

        virtual std::ostream& Dump(std::ostream&) const;

    protected:
        void ContractLocal(int ia, const Matrix4T& Hlocal, ArrayCT& newAmplitudes) const;


    private:
        int      itsL;
        double   itsS;
        int      itsp;
        long int itsN; //p^L
        double   itsE; //Eigen energy from power iterations
        ArrayCT itsAmplitudes;
};

#endif // FULLSTATEIMP_H
