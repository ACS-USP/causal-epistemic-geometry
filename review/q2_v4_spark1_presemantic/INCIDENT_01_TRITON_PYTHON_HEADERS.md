# Q2 V4 incident 01 — missing Triton build headers

Classification: `CLASS_A_DEPENDENCY_RESTORATION`, pre-outcome.

After the Spark execution guard was prospectively amended, the exact model loaded but the
first technical fixture stopped before producing an output. PyTorch 2.13's native eager
router invoked Triton, whose small CUDA helper could not compile because `Python.h` was
absent. No technical trajectory, source output, activation archive, candidate bank, shell
result, geometry value, or semantic outcome was produced.

The compatible Ubuntu ARM64 packages `python3.12-dev=3.12.3-1ubuntu0.16` and
`libpython3.12-dev=3.12.3-1ubuntu0.16` were downloaded and extracted under the user's
project directory without `sudo`, global package installation, OS upgrade, CUDA change,
or driver change. Package SHA-256 values:

- `python3.12-dev`: `424a323ebfacc1454c805cb3486525b73dfb684d21f548ccfb96ba0eaf3a96a8`
- `libpython3.12-dev`: `644c40641fc40da4f8205fb96b60b404677e28402d8b2b55d437697cae1d7f96`

The runner records the effective include path and `Python.h` hash in the environment lock.
The scientific protocol is unchanged.
