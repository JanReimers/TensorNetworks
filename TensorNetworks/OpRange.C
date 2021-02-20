#include "TensorNetworks/SiteOperator.H"
#include "oml/imp/matlimit.h"

namespace TensorNetworks
{

void OpRange::resize(const MatLimits& l)
{
    rows.clear();
    cols.clear();
    rows.resize(l.GetNumCols(),Range({l.Row.High,l.Row.Low-1})); //Pessimistic starting point
    cols.resize(l.GetNumRows(),Range({l.Col.High,l.Col.Low-1}));
    row=Range({l.Row.Low,l.Row.High});
    col=Range({l.Col.Low,l.Col.High});
    Dw1=l.GetNumRows();
    Dw2=l.GetNumCols();
}
void OpRange::NonZeroAt(index_t i,index_t j)
{
    if (rows[j].Low >i) rows[j].Low =i;
    if (rows[j].High<i) rows[j].High=i;
    if (cols[i].Low >j) cols[i].Low =j;
    if (cols[i].High<j) cols[i].High=j;
}

std::ostream& operator<<(std::ostream& os,const OpRange& r)
{
    os << "Row.Low =";
    for (auto i:  r.rows) os << i.Low << " ";
    os << std::endl;
    os << "Row.High=";
    for (auto i:  r.rows) os << i.High << " ";
    os << std::endl;
    os << "Col.Low =";
    for (auto i:  r.cols) os << i.Low << " ";
    os << std::endl;
    os << "Col.High=";
    for (auto i:  r.cols) os << i.High << " ";
    os << std::endl;
    return os;
}

}
