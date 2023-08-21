# Triereme

We introduce Triereme, a method to speed up concolic engines in hybrid fuzzers
by reducing the time spent in the SMT solver. Triereme schedules and caches
solver queries using a trie (or prefix tree) data structure, thereby making it
easier to exploit common prefixes. This design is made possible by decoupling
concolic tracing from concolic solving, running both in separate processes. As a
result, Triereme manages to reconcile pruning through path constraint filtering
with incremental solving, allowing it to reap their combined benefits.

Our prototype borrows the instrumentation from [SymCC][symcc] and relies on
[LibAFL][libafl] for its fuzzing helper. The implementation is heavily inspired
by [QSYM][qsym].

A thorough description of this work can be found in "Triereme: Speeding up
hybrid fuzzing through efficient query scheduling", conditionally accepted at
ACSAC 2023.

The FuzzBench fork used for our evaluation can be found
[here][fuzzbench-triereme].

[fuzzbench-triereme]: https://github.com/vusec/fuzzbench-triereme
[symcc]: https://github.com/eurecom-s3/symcc
[libafl]: https://github.com/AFLplusplus/LibAFL
[qsym]: https://github.com/sslab-gatech/qsym


## Building Triereme

Detailed building instructions for Ubuntu 20.04 can be found in the
[Dockerfile][triereme-build] within the FuzzBench fork used for our evaluation.

This file includes information regarding the dependecies required, how to build
AFL++, and how to build libcxx and libcxx-abi for C++ support.

[triereme-build]: https://github.com/vusec/fuzzbench-triereme/blob/triereme/fuzzers/triereme_trie_single/builder.Dockerfile


## Running Triereme

### Build Target Program

The target program needs to be built twice: once with the AFL++ instrumentation
and one with the SymCC instrumentation. The AFL++ binary can be customized
independently with a variety of options that are described in the [original
documentation][aflpp-docs]; we recommend using at least CmpLog and non-colliding
coverage. Detailed instructions on the appropriate flags and environment
variables that need to be used to build the program can be found in the [build
scripts][target-build] contained in the FuzzBench fork that was used for our
evaluation.

[aflpp-docs]: https://aflplus.plus/features/
[target-build]: https://github.com/vusec/fuzzbench-triereme/blob/8e181c1daf1e8b3590b59f649ee622e6d334ec56/fuzzers/symcc_libafl_single/fuzzer.py#L30-L69


### Fuzzing

Once the two instrumented versions of the target program have been built, AFL++
and the concolic engine have to be started separately. Precise instructions on
how to start both components can be found in the corresponding
[script][target-run] in our FuzzBench fork. As before, the options used for
AFL++ can be freely customized following the original documentation.

[target-run]: https://github.com/vusec/fuzzbench-triereme/blob/triereme/fuzzers/triereme_trie_single/fuzzer.py#L27-L63
