# Third-Party Notices

This distribution includes code adapted from the following open-source
projects. Copyright notices in the source files must be retained.

## FlashAttention

Repository: <https://github.com/Dao-AILab/flash-attention>

- `vllm/models/step5/nvidia/ops/cute_dsl/cute_dsl_utils.py` derives from
  `flash_attn/cute/cute_dsl_utils.py` at
  `0256114fe2381ab293503219bdd9078de3cd26b3`.
- `vllm/models/step5/nvidia/ops/cute_dsl/flash_attn/copy_utils.py` derives
  from `flash_attn/cute/copy_utils.py` at
  `0256114fe2381ab293503219bdd9078de3cd26b3`.
- `vllm/models/step5/nvidia/ops/cute_dsl/flash_attn/utils.py` derives from
  `flash_attn/cute/utils.py` at
  `0256114fe2381ab293503219bdd9078de3cd26b3`.
- `vllm/models/step5/nvidia/ops/cute_dsl/sparse_gqa/kernel_utils/`
  `mma_helpers.py`, `hopper_helpers.py`, and `softmax.py` contain code derived
  from `flash_attn/cute/ampere_helpers.py`, `hopper_helpers.py`, and
  `softmax.py`, respectively, at
  `0256114fe2381ab293503219bdd9078de3cd26b3`.
- `vllm/models/step5/nvidia/ops/cute_dsl/sparse_gqa/kernel_utils/`
  `cute_utils.py` contains code derived from `flash_attn/cute/utils.py` and
  `flash_attn/cute/copy_utils.py` at
  `0256114fe2381ab293503219bdd9078de3cd26b3`.

The FlashAttention-derived portions of these files are licensed under the
BSD 3-Clause License. The license text is
provided in `THIRD_PARTY_LICENSES/flash-attention-BSD-3-Clause.txt`; the
referenced contributor list is provided in
`THIRD_PARTY_LICENSES/flash-attention-AUTHORS`.

## QuACK

Repository: <https://github.com/Dao-AILab/quack>

- `vllm/models/step5/nvidia/ops/cute_dsl/reduction_base.py` derives from
  `quack/reduction_base.py` at
  `2d98b796a3e04e31c433b5fc40daf14954c4da09`.
- `vllm/models/step5/nvidia/ops/cute_dsl/utils.py` derives from
  `quack/utils.py` at `ec2980f20df49b7eb1190123eabbaa414713e35f`.
- `vllm/models/step5/nvidia/ops/cute_dsl/sparse_gqa/kernel_utils/`
  `cute_utils.py` contains code derived from `quack/utils.py` at
  `2d98b796a3e04e31c433b5fc40daf14954c4da09`.

The QuACK-derived portions of these files are licensed under Apache-2.0. The
repository-level `LICENSE` contains the applicable Apache License, Version 2.0
text.
