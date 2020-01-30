
%MappedType std::string
{
%TypeHeaderCode
#include <string>
%End


%ConvertToTypeCode
    if (sipIsErr == NULL)
    {
        if (!PyUnicode_Check(sipPy))
            return 0;
        else
            return 1;
    }

    std::string *s = new std::string(PyUnicode_AsUTF8(sipPy));
    *sipCppPtr = s;

    return sipGetState(sipTransferObj);
%End

%ConvertFromTypeCode
    PyObject *s = PyUnicode_FromString(sipCpp->c_str());

    return s;
%End
};
