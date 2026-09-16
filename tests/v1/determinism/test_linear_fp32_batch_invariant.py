# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from utils import skip_unsupported

from vllm.model_executor.layers.batch_invariant import (
    linear_fp32_batch_invariant,
)


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_linear_fp32_output_is_batch_invariant(default_vllm_config, dtype):
    torch.manual_seed(8193)
    weight = torch.randn((352, 4096), device="cuda", dtype=dtype)
    batched_input = torch.randn((257, 4096), device="cuda", dtype=dtype)

    batched_output = linear_fp32_batch_invariant(batched_input, weight)

    assert batched_output.dtype == torch.float32
    for row in (0, 42, 128, 256):
        single_output = linear_fp32_batch_invariant(
            batched_input[row : row + 1], weight
        )
        torch.testing.assert_close(
            single_output[0], batched_output[row], rtol=0, atol=0
        )


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_linear_fp32_output_matches_reference(default_vllm_config, dtype):
    torch.manual_seed(8193)
    weight = torch.randn((352, 4096), device="cuda", dtype=dtype)
    input_tensor = torch.randn((17, 4096), device="cuda", dtype=dtype)

    output = linear_fp32_batch_invariant(input_tensor, weight)
    reference = torch.nn.functional.linear(input_tensor.float(), weight.float())

    torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-2)
