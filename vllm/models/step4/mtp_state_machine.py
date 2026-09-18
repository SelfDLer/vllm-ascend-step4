# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step4 dispatch from vLLM request metadata.

Cross-step MTP state lives in the GPU transaction buffers in sparse_attention_mtp.
This plan describes only the current forward; it never mirrors sampler state.
"""

from dataclasses import dataclass

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills


@dataclass(frozen=True)
class Step4KernelPlan:
    """One backend-local plan shared by attention and summary updates."""

    num_decode_requests: int
    num_decode_tokens: int
    has_q1_decode: bool
    has_verifier: bool
    all_decode_q4_or_padding: bool
    use_flattened_decode: bool


def build_step4_kernel_plan(
    metadata: CommonAttentionMetadata,
    *,
    decode_threshold: int,
    mtp_enabled: bool,
) -> Step4KernelPlan:
    """Use the upstream decode/prefill split, including all ordinary q1 rows.

    Prefill lifecycle takes priority over speculative-looking query lengths.
    Slot mapping and GPU seq_lens still determine which rows are valid; neither
    padding nor speculative acceptance can be inferred from host sequence lengths.
    """
    num_reqs = metadata.num_reqs
    starts = metadata.query_start_loc_cpu
    if (
        starts.device.type != "cpu"
        or starts.dtype != torch.int32
        or starts.ndim != 1
        or starts.numel() != num_reqs + 1
        or int(starts[0]) != 0
    ):
        raise ValueError("Step4 requires CPU int32 query boundaries for every request")
    query_lens = starts[1:] - starts[:-1]
    if bool((query_lens < 0).any()) or int(starts[-1]) > metadata.num_actual_tokens:
        raise ValueError("Step4 query boundaries exceed the flattened token range")
    if num_reqs == 0:
        return Step4KernelPlan(
            num_decode_requests=0,
            num_decode_tokens=0,
            has_q1_decode=False,
            has_verifier=False,
            all_decode_q4_or_padding=False,
            use_flattened_decode=False,
        )

    num_decodes, _, num_decode_tokens, _ = split_decodes_and_prefills(
        metadata,
        decode_threshold=decode_threshold,
        treat_short_extends_as_decodes=False,
    )
    # The kernels consume contiguous request prefixes. Reject an interleaved
    # lifecycle instead of silently treating a later verifier as prefill.
    is_decode = (query_lens > 0) & (query_lens <= decode_threshold)
    if metadata.is_prefilling is not None:
        is_decode &= ~metadata.is_prefilling[:num_reqs]
    if bool(is_decode[num_decodes:].any()):
        raise ValueError("Step4 decode requests must precede prefill requests")
    decode_lens = query_lens[:num_decodes]
    has_verifier = mtp_enabled and bool((decode_lens > 1).any())
    if bool((decode_lens > 4).any()):
        raise ValueError("Step4 DSA decode kernels support at most four query rows")
    return Step4KernelPlan(
        num_decode_requests=num_decodes,
        num_decode_tokens=num_decode_tokens,
        has_q1_decode=bool((decode_lens == 1).any()),
        has_verifier=has_verifier,
        all_decode_q4_or_padding=bool(((decode_lens == 4) | (decode_lens == 0)).all()),
        use_flattened_decode=(
            num_decode_tokens > 0 and not bool((query_lens[num_decodes:] > 0).any())
        ),
    )
