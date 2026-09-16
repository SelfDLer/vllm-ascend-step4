# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.step4.mtp_state_machine import build_step4_kernel_plan


def _plan(query_lens, is_prefilling, *, mtp=True, padded_tokens=0):
    starts = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    return build_step4_kernel_plan(
        SimpleNamespace(
            query_start_loc_cpu=starts,
            num_reqs=len(query_lens),
            num_actual_tokens=sum(query_lens) + padded_tokens,
            max_query_len=max(query_lens, default=0),
            is_prefilling=torch.tensor(is_prefilling, dtype=torch.bool),
        ),
        decode_threshold=4 if mtp else 1,
        mtp_enabled=mtp,
    )


@pytest.mark.parametrize(
    "queries,prefilling,prefix,tokens,q1,verifier,flat",
    [
        ([], [], 0, 0, False, False, False),
        ([4, 1, 8], [False, False, True], 2, 5, True, True, False),
        ([1, 4, 8], [False, False, True], 2, 5, True, True, False),
        ([1, 4], [False, True], 1, 1, True, False, False),
        ([4], [True], 0, 0, False, False, False),
        ([1], [True], 0, 0, False, False, False),
        ([4, 4], [False, False], 2, 8, False, True, True),
        ([1, 1, 0], [False, False, False], 3, 2, True, False, True),
        ([0, 0], [False, False], 2, 0, False, False, False),
        # Graph warmup has no request lifecycle but keeps the structural split.
        ([1, 9], [False, False], 1, 1, True, False, False),
    ],
)
def test_plan_uses_upstream_lifecycle_and_query_boundaries(
    queries, prefilling, prefix, tokens, q1, verifier, flat
):
    plan = _plan(queries, prefilling)
    assert (plan.num_decode_requests, plan.num_decode_tokens) == (prefix, tokens)
    assert (plan.has_q1_decode, plan.has_verifier) == (q1, verifier)
    assert plan.use_flattened_decode == flat


def test_graph_token_capacity_does_not_change_decode_dispatch():
    assert _plan([1, 1], [False, False], padded_tokens=6).use_flattened_decode


def test_interleaved_verifier_is_rejected_before_summary_mutation():
    with pytest.raises(ValueError, match="must precede"):
        _plan([4, 4], [True, False])


def test_non_mtp_short_prefill_keeps_ordinary_update():
    plan = _plan([1, 2], [False, True], mtp=False)
    assert plan.num_decode_tokens == 1
    assert not plan.has_verifier
    assert not plan.use_flattened_decode
