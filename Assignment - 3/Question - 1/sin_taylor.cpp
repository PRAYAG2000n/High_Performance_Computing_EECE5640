#include <iostream>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <string>
#include <cstring>

/**
 * Compute sin(x) using the first 'terms' terms of its Taylor series:
 *   sin(x) = x - x^3/3! + x^5/5! - ...
 * This version uses 'float' internally (single precision).
 */
float sinTaylorFloat(float x, int terms = 10)
{
    float sum   = 0.0f;
    float term  = x;       // First term = x^(2*0+1)/(2*0+1)! = x
    int   sign  = 1;       // Alternates +1, -1, +1, ...

    for (int n = 0; n < terms; n++)
    {
        sum += (sign * term);

        // Next term: multiply current term by x^2 / [(2n+2)*(2n+3)]
        // Be careful with integer->float to ensure no type issues
        float denom = float((2*(n+1))*(2*(n+1)+1));
        term = term * (x * x) / denom;

        sign = -sign;
    }
    return sum;
}

/**
 * Compute sin(x) using double precision.
 */
double sinTaylorDouble(double x, int terms = 10)
{
    double sum  = 0.0;
    double term = x;    // first term = x
    int    sign = 1;

    for (int n = 0; n < terms; n++)
    {
        sum += (sign * term);

        double denom = double((2*(n+1)) * (2*(n+1)+1));
        term = term * (x * x) / denom;

        sign = -sign;
    }
    return sum;
}

/**
 * Convert a float (single precision) to its 32-bit IEEE 754 bitstring.
 *
 * Approach:
 *   1. Reinterpret the float bits as a 32-bit integer (uint32_t).
 *   2. Print in binary.
 */
std::string floatToBinary32(float f)
{
    // Reinterpret the memory of 'f' as an unsigned 32-bit integer
    static_assert(sizeof(float) == sizeof(uint32_t), "Size mismatch");
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    // Build a 32-char string of bits
    std::string result;
    result.reserve(32);
    for (int i = 31; i >= 0; i--)
    {
        result.push_back((bits & (1u << i)) ? '1' : '0');
    }
    return result;
}

/**
 * Convert a double (double precision) to its 64-bit IEEE 754 bitstring.
 */
std::string doubleToBinary64(double d)
{
    // Reinterpret the memory of 'd' as an unsigned 64-bit integer
    static_assert(sizeof(double) == sizeof(uint64_t), "Size mismatch");
    uint64_t bits;
    std::memcpy(&bits, &d, sizeof(bits));

    // Build a 64-char string of bits
    std::string result;
    result.reserve(64);
    for (int i = 63; i >= 0; i--)
    {
        result.push_back((bits & (1ULL << i)) ? '1' : '0');
    }
    return result;
}

int main()
{
    // Part (a): Evaluate sin(x) using Taylor expansions
    const float  xs_float[4]  = {0.1f, 1.0f, 3.14f, 6.28f};
    const double xs_double[4] = {0.1,  1.0,  3.14,  6.28};

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "Comparison of sin(x) with 10-term Taylor expansion:\n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "   x       float_Taylor        double_Taylor       std::sin(x)\n";
    std::cout << "---------------------------------------------------\n";

    for (int i = 0; i < 4; i++)
    {
        float  x_f = xs_float[i];
        double x_d = xs_double[i];

        float  val_f  = sinTaylorFloat(x_f, 10);
        double val_d  = sinTaylorDouble(x_d, 10);
        double val_std = std::sin(x_d); // reference from <cmath>

        std::cout << std::setw(5) << x_f << "   "
                  << std::setw(15) << val_f  << "   "
                  << std::setw(15) << val_d  << "   "
                  << std::setw(15) << val_std << "\n";
    }
    std::cout << "\n";


    // Part (b): Print IEEE 754 single/double-precision representations
    double values[3] = {2.1, 6300.0, -1.044};

    std::cout << "IEEE 754 Representations:\n";
    std::cout << "-------------------------\n";
    for (double v : values)
    {
        float  vf = static_cast<float>(v);
        std::string bin32 = floatToBinary32(vf);
        std::string bin64 = doubleToBinary64(v);

        std::cout << "Value = " << v << "\n";
        std::cout << "  Single Precision (32 bits): " << bin32 << "\n";
        std::cout << "  Double Precision (64 bits): " << bin64 << "\n\n";
    }

    return 0;
}
