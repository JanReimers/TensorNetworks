// File: VNEntropy.H  Calculate the Von Neumann entropy from a list ofdensity matrix diagonals or from a list of singular values

template <class T> class Vector;

// From density matrix eigen values
// Sum(Si*ln(Si))
double VNEntropyFromEVs(const Vector<double>& s);
// From singular values
// Sum(Si^2*ln(Si^2))
double VNEntropyFromSVs(const Vector<double>& s);

