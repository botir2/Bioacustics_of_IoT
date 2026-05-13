#ifndef FFTHELPER_H
#define FFTHELPER_H

#include <complex>
#include <vector>

class FftHelper
{
public:
    static void fft(std::vector<std::complex<double>> &data);
};

#endif
