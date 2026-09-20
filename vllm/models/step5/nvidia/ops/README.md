# Step5 NVIDIA operators

This directory contains the NVIDIA-only inference kernel closure required by
the Step5 model:

- `cute_dsl/`: QK/indexer norm + RoPE, DSA indexer/summary kernels, sparse GQA
  prefill/decode, and MTP sparse-attention kernels;
- `triton/`: the Step5 router-bias kernel and its driver-launch helpers.

It intentionally excludes training, backward, benchmark, experimental,
DeepGEMM/MoE, quantization, and unrelated model kernels. Native
libtorch-stable CUDA operators remain under
`csrc/libtorch_stable/step5/`.

The Python package roots are lazy and must not initialize CUDA during model
registry import.

Source files in this directory must retain their authoritative copyright and
SPDX license declarations. Known third-party sources and their required
license materials are listed in the repository-level
`THIRD_PARTY_NOTICES.md`. Any source file without a verified redistribution
grant and accurate license declaration must be cleared by its rightsholder
before a public source distribution or wheel is published.
