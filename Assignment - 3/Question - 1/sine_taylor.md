
# C++ Implementation of Sin Taylor Series with IEEE 754 Compliance

## Project Overview
This project offers a C++ implementation of the Taylor series approximation for the sine function, `sin(x)`, focusing on adherence to the IEEE 754 standard for floating-point arithmetic. The goal is to explore the implications of using different floating-point precisions (single and double) in computational accuracy and performance.

## Features
- C++ implementation using Taylor series to compute `sin(x)`.
- Utilization of IEEE 754 single and double precision floating points.
- Detailed comparison of results between single precision and double precision.

## How to Compile and Run
### Prerequisites
- A modern C++ compiler that supports C++11 or later (e.g., GCC, Clang).
- Make sure the compiler supports IEEE 754 floating-point operations.

### Create a C++ file in Rocky Linux
```bash
Command line: nano sin_taylor.cpp
```
### Compilation
Use the following command to compile the program, ensuring that IEEE 754 support is enabled:
```bash
g++ -std=c++11 -o sin_taylor sin_taylor.cpp -lm
```

### Execution
Run the compiled program:
```bash
./sin_taylor
```

## Implementation Details
The program computes the sine of a given angle (in radians) using the Taylor series expansion:
sin(x) = x - x^3/3! + x^5/5! - x^7/7! + x^9/9! + ......
             
The user can specify the number of terms to use in the series, which demonstrates how increasing the number of terms can improve the approximation's accuracy.

## IEEE 754 Compliance
The IEEE 754 standard is used to ensure consistent floating-point computation. This implementation uses:
- **Single Precision (32-bit)**: Provides about 7 decimal digits of precision.
- **Double Precision (64-bit)**: Provides about 15 decimal digits of precision.

## Understanding Floating-Point Precision
The comparison of single and double precision highlights:
- The rounding errors inherent in binary floating-point representation.
- The impact of precision on the accuracy of mathematical computations.

## Contributions
This project is open to contributions, especially those that might improve computational efficiency, enhance precision handling, or extend the educational aspects of floating-point arithmetic.

## Sources
https://stackoverflow.com/questions/22416710/in-c-finding-sinx-value-with-taylors-series
https://cplusplus.com/forum/beginner/146496/
https://github.com/drewtu2/eece5640/blob/master/hw3/q1/src/main.cpp
