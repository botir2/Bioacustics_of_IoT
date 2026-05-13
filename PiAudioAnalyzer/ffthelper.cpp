#include "ffthelper.h"

#include <algorithm>
#include <cmath>

void FftHelper::fft(std::vector<std::complex<double>> &data)
{
    int n = static_cast<int>(data.size());

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;

        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }

        j ^= bit;

        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    const double pi = std::acos(-1.0);

    for (int len = 2; len <= n; len <<= 1) {
        double angle = -2.0 * pi / len;
        std::complex<double> wlen(std::cos(angle), std::sin(angle));

        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);

            for (int j = 0; j < len / 2; j++) {
                std::complex<double> u = data[i + j];
                std::complex<double> v = data[i + j + len / 2] * w;

                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;

                w *= wlen;
            }
        }
    }
}
