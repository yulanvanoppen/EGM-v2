# EGM: Extended Gradient Matching
_MATLAB® code package for parameter inference of ODE models with uncertain initial conditions as presented in [1]_

_Updated version of [yulanvanoppen/EGM](https://github.com/yulanvanoppen/EGM)_

&nbsp;

## Quickstart
Install MATLAB® R2023b with IQM Tools Pro Version 1.2.2 (02.01.2017) and Monolix 2021R2.

Add folder `EGM/` to path and run `example.m`.

Run to reproduce outcomes:

-`accuracy_HS.m` and `accuracy_GLV.m` for Figures 2, 3, and 4 (first two panels);

-`test_TCS.m`, `test_repressilator.m`, and `test_STAT5.m` for Figures 4 (last panel), 5;

-`experimental_STAT5.m` for Figure 6;

-`convergence_HS.m` and `convergence_GLV.m` to reproduce the results in Figures 9, 10.

&nbsp;

## References
[1] van Oppen, Yulan B. and Milias-Argeitis, Andreas. A new gradient matching method for fast ODE inference. _Manuscript in preparation._

&nbsp;

## DISCLAIMER
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
