---
title: The 'matrix-multiply-pthread-longrun' binary
layout: default
permalink: resources/matrix-multiply-pthread-longrun
shortdoc: >
    Source for 'matrix-multiply-pthread-longrun'. A pthread matrix-multiply binary runs a multiplication on two 150x150 matrixes using persistent worker threads. The sum of the multiplied matrix is printed upon completion.
author: ["Bobby R. Bruce"]
---

The 'matrix-multiply-pthread-longrun' resource runs a multiplication on two
150x150 matrixes.
The multiplication is parallelized with persistent pthread workers, and the
main thread also participates as a compute worker. The number of iterations
and the number of worker threads are passed on the command line.
The sum of the multiplied matrix is printed upon completion.

## Building Instructions

Run `make`.

This will only compile the binary to the X86 ISA.

## Cleaning Instructions

Run `make clean` in the Makefile directory.

## Usage

As this binary does not contain any special `m5` library code it can be run outside of a gem5 simulation:

```sh
./matrix-multiply-longrun <iterations> <num_workers>
```

Example for a 4-core CPU:

```sh
./matrix-multiply-longrun 1000 3
```

It can also be run in a gem5 simulation in SE mode.
Below is a snippet which utilizes the gem5 standard library to do so:

```py
board.set_se_workload(Resource("x86-matrix-multiply-longrun"))

simulator = Simulator(board = board)
```

## Pre-built binaries

Compiled to the X86 ISA: http://dist.gem5.org/dist/develop/test-progs/matrix-multiply-longrun/x86-matrix-multiply-longrun

## License

This code is covered by By the [03-Clause BSD License (BSD-3-Clause)](https://opensource.org/licenses/BSD-3-Clause).
