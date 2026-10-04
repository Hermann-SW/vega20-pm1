# vega20-pm1
IBDWT p-1 proof for primes on high FP64 performance AMD GPUs like Instinct MI50/Radeon Pro VII/Radeon VII with 6.7/6.5/3.5 TFLOPS FP64. Developed in very long chat with free gemini.google.com (>10h). Work in progress ...

Developed first with primorial primes (p#+1) in mind, p-1 proof works for all numbers p with known factorization of p-1. This includes big Carmichael numbers created from big smooth numbers and Proth numbers p=k\*2^n+1 with $k\leq n$.

CPU side verification part runs long at 796% CPU on 4C/8T Intel Xeon W-2225 CPU.
Only the first two pimorial prime testcases fit into current 524288 element restriction:  
```
hermann@Radeon-pro-vii:~/vega20-pm1$ f=ibdwt_2d_verification_harness
hermann@Radeon-pro-vii:~/vega20-pm1$ hipcc -O3 -std=c++20 --offload-arch=gfx906 -ffp-contract=fast -fopenmp $f.cpp -o $f
hermann@Radeon-pro-vii:~/vega20-pm1$ /usr/bin/time ./ibdwt_2d_verification_harness 
=======================================================
--- Item 1 Test: Impulse Response Vector ---
=======================================================
--- Spectrum Sample (Forward) ---
  [0] = (-212484, -977.846i)
  [1] = (-70826.9, -543.246i)
  [2] = (-42495.2, -456.325i)
  [3] = (-30352.9, -419.073i)
  [4] = (-23607, -398.377i)
  [5] = (-19314, -385.206i)
  [6] = (-16341.9, -376.088i)
  [7] = (-14162.2, -369.401i)

GPU Output for Impulse Input [1.0, 1.0, ...]:
  Bin[0] = 1.000000
  Bin[1] = 1.000000
  Bin[2] = 1.000000
  Bin[3] = 1.000000

=======================================================
--- Running Identity Round-Trip ---
=======================================================

Max Measured FP64 Roundoff Error: 7.2759576e-12 [PASS]

=======================================================
--- Running Full Convolution (Square) ---
=======================================================

  --- Sample Comparison (GPU vs CPU Exact 2D Negacyclic) ---
  Bin[0] GPU: 143584243848704.00 | CPU Exact: 143584243848704.00 | Diff: 0.00e+00
  Bin[1] GPU: 149713076870144.00 | CPU Exact: 149713076870144.00 | Diff: 0.00e+00
  Bin[2] GPU: 154410674540420.00 | CPU Exact: 154410674540420.00 | Diff: 0.00e+00
  Bin[3] GPU: 157706562411023.97 | CPU Exact: 157706562411024.00 | Diff: 3.12e-02
  Bin[4] GPU: 159615233647912.00 | CPU Exact: 159615233647912.00 | Diff: 0.00e+00
  Bin[5] GPU: 160151181417039.97 | CPU Exact: 160151181417040.00 | Diff: 3.12e-02
  Bin[6] GPU: 159328898884363.97 | CPU Exact: 159328898884364.00 | Diff: 3.12e-02
  Bin[7] GPU: 157125835122912.03 | CPU Exact: 157125835122912.00 | Diff: 3.12e-02

=======================================================
--- CONVOLUTION ACCURACY ANALYSIS (524288 bins) ---
=======================================================
Max Absolute FP64 Error : 8.5937500e-02 at Index [38243]
RMS Error               : 2.3420783e-02
Max Relative Error      : 1.9911572e-09
Exact Threshold Margin  : 0.4141 (0.5 rounding boundary)
Overall Status          : [PASS] Safe for Rounding
=======================================================


Running target: 13649# + 1 (N = 19475)

Running target: 42209# + 1 (N = 60595)

Skipping target 4328927# + 1 (requires 6241400 elements, harness max is 524288)

Skipping target 9562633# + 1 (requires 13791000 elements, harness max is 524288)
495.97user 0.07system 1:02.31elapsed 796%CPU (0avgtext+0avgdata 168204maxresident)k
0inputs+12384outputs (1major+17634minor)pagefaults 0swaps
hermann@Radeon-pro-vii:~/vega20-pm1$ 
```
