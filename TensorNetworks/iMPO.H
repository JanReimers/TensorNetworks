#ifndef iMPO_H
#define iMPO_H

#include "TensorNetworks/MPO.H"

namespace TensorNetworks
{
//
//  MPO is an operator but it also has the combine and compress capabilities.
//  Most of the this interface can be implemented using a few abstract functions
//  from the Operator class.
//
class iMPO : public virtual MPO
{
    public:
        iMPO() {};
        virtual ~iMPO() {};

        virtual iMPO* MakeUnitcelliMPO(int unitcell) const=0; //Contract horizontally to make iMPO for the whole unit cell.

        virtual int   GetL() const=0;
        virtual const SiteOperator* GetSiteOperator(int isite) const
        {
            return MPO::GetSiteOperator(isite);
        }
        virtual       SiteOperator* GetSiteOperator(int isite)=0;
};

}

#endif // iMPO_H
