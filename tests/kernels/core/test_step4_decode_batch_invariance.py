import math

import pytest
import torch

from vllm.models.step4.nvidia.ops.cute_dsl.sparse_gqa import (
    merge_dynamic_split_nat_lse_states_sm90_gqa,
    merge_variable_split_nat_lse_states_sm90_gqa,
    token_wise_flash_attn_decode_sm90_gqa_func,
    token_wise_flash_attn_decode_sm90_gqa_plan,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="Step4 SM90 decode requires Hopper",
)


def make_case(regions=512, batch=4, head_dim=128):
    torch.manual_seed(20260906)
    query = torch.randn(4, 16, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(
        (regions + 1) // 2, 16, 1, head_dim, device="cuda", dtype=query.dtype
    )
    value = torch.randn_like(key)
    query = query.repeat((math.ceil(batch / 4), 1, 1))[:batch].contiguous()
    counts = torch.full((batch,), regions, device="cuda", dtype=torch.int32)
    packed = torch.full((batch, 513), -1, device="cuda", dtype=torch.int64)
    region_ids = torch.arange(regions, device="cuda", dtype=torch.int64)
    packed[:, :regions] = (region_ids * 8) | (region_ids << 32)
    seq_lens = torch.full((batch,), regions * 8, device="cuda", dtype=torch.int32)
    return query, key, value, counts, packed, seq_lens


def decode(case, query_len, invariant=True, starts=None, valid_rows=None, split_max=16):
    query, key, value, counts, packed, seq_lens = case
    batch, heads, head_dim = query.shape
    if split_max in (2, 4):
        padded_columns = math.ceil(packed.shape[1] / split_max) * split_max
        padded = packed.new_full((batch, padded_columns), -1)
        padded[:, : packed.shape[1]] = packed
        packed = padded
    plan = token_wise_flash_attn_decode_sm90_gqa_plan(
        batch,
        dtype=query.dtype,
        q_heads=heads,
        head_dim=head_dim,
        head_dim_v=head_dim,
        topk_windows=packed.shape[1],
        variable_split_max=split_max,
        mtp_q_len=query_len,
        batch_invariant=invariant,
        sm_count=torch.cuda.get_device_properties(query.device).multi_processor_count,
    )
    partial = torch.empty(plan[2], heads, head_dim, device="cuda", dtype=query.dtype)
    partial_lse = torch.empty(plan[2], heads, device="cuda", dtype=torch.float32)
    output = torch.empty_like(query)
    lse = torch.empty(batch, heads, device="cuda", dtype=torch.float32)
    if query_len > 1 and starts is None:
        starts = torch.arange(0, batch + 1, query_len, device="cuda", dtype=torch.int32)
    token_wise_flash_attn_decode_sm90_gqa_func(
        query,
        key,
        value,
        region_counts=counts,
        region_packed_indices=packed,
        kv_seqlens=seq_lens,
        out=partial,
        lse=partial_lse,
        softmax_scale=1 / math.sqrt(head_dim),
        variable_split_max=split_max,
        mtp_q_len=query_len,
        batch_invariant=invariant,
        query_start_loc=starts,
        valid_rows=valid_rows,
    )
    if split_max == 16:
        merge_dynamic_split_nat_lse_states_sm90_gqa(
            partial,
            partial_lse,
            counts,
            output,
            lse,
            mtp_q_len=query_len,
            batch_invariant=invariant,
        )
    else:
        merge_variable_split_nat_lse_states_sm90_gqa(
            partial, partial_lse, output, lse, n_split4=plan[0], n_split2=plan[1]
        )
    return output, lse, plan


@pytest.mark.parametrize("regions", [511, 512, 513])
@pytest.mark.parametrize("head_dim", [128, 192])
def test_grouping_preserves_outputs_and_reference(regions, head_dim):
    case = make_case(regions, head_dim=head_dim)
    ordinary, ordinary_lse, _ = decode(case, 1)
    speculative, speculative_lse, _ = decode(case, 4)
    assert torch.equal(ordinary, speculative)
    assert torch.equal(ordinary_lse, speculative_lse)
    query, key, value, *_ = case
    scores = torch.einsum(
        "bhd,nd->bhn", query.float(), key.reshape(-1, head_dim)[: regions * 8].float()
    ) / math.sqrt(head_dim)
    reference = torch.einsum(
        "bhn,nd->bhd",
        scores.softmax(-1),
        value.reshape(-1, head_dim)[: regions * 8].float(),
    )
    torch.testing.assert_close(ordinary.float(), reference, rtol=0.02, atol=0.002)


@pytest.mark.parametrize("batch", [1, 4, 32, 160])
def test_unrelated_requests_do_not_change_numerical_partition(batch):
    baseline, baseline_lse, _ = decode(make_case(batch=1), 1)
    output, lse, plan = decode(make_case(batch=batch), 1)
    assert plan[2] == batch * 16
    assert torch.equal(baseline[0], output[0])
    assert torch.equal(baseline_lse[0], lse[0])


def test_policy_cache_does_not_reuse_default_grouping_variant():
    case = make_case()
    ordinary, _, _ = decode(case, 1, invariant=False)
    adaptive, _, _ = decode(case, 4, invariant=False)
    invariant, _, _ = decode(case, 4, invariant=True)
    adaptive_again, _, _ = decode(case, 4, invariant=False)
    assert not torch.equal(ordinary, adaptive)
    assert torch.equal(ordinary, invariant)
    assert torch.equal(adaptive, adaptive_again)


@pytest.mark.parametrize("query_len", [2, 3])
def test_short_verifier_groups_preserve_target_outputs(query_len):
    case = make_case(batch=12, head_dim=192)
    ordinary, ordinary_lse, _ = decode(case, 1)
    speculative, speculative_lse, _ = decode(case, query_len)
    assert torch.equal(ordinary, speculative)
    assert torch.equal(ordinary_lse, speculative_lse)


@pytest.mark.parametrize("boundaries", [[0, 1, 4, 6, 8], [0, 3, 7]])
def test_ragged_grouping_preserves_flat_row_outputs(boundaries):
    case = make_case(batch=boundaries[-1])
    starts = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    ordinary, ordinary_lse, _ = decode(case, 1)
    speculative, speculative_lse, _ = decode(case, 4, starts=starts)
    assert torch.equal(ordinary, speculative)
    assert torch.equal(ordinary_lse, speculative_lse)


@pytest.mark.parametrize("split_max", [2, 4])
def test_fixed_split_modes_do_not_change_with_batch_size(split_max):
    ordinary, ordinary_lse, small_plan = decode(
        make_case(batch=1), 1, split_max=split_max
    )
    batched, batched_lse, large_plan = decode(
        make_case(batch=160), 1, split_max=split_max
    )
    assert small_plan[2] == split_max
    assert large_plan[2] == 160 * split_max
    assert torch.equal(ordinary[0], batched[0])
    assert torch.equal(ordinary_lse[0], batched_lse[0])


def test_graph_padding_and_replay_preserve_live_rows():
    case = make_case(batch=16)
    counts = case[3]
    counts[4:] = 0
    valid_rows = torch.tensor([4], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 1, 4, 4, 4], device="cuda", dtype=torch.int32)
    decode(case, 4, starts=starts, valid_rows=valid_rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, lse, _ = decode(case, 4, starts=starts, valid_rows=valid_rows)
    counts[3:] = 0
    valid_rows.fill_(3)
    starts.copy_(torch.tensor([0, 1, 3, 3, 3], device="cuda", dtype=torch.int32))
    output.fill_(float("nan"))
    graph.replay()
    reference, reference_lse, _ = decode(case, 1, valid_rows=valid_rows)
    assert torch.equal(output[:3], reference[:3])
    assert torch.equal(lse[:3], reference_lse[:3])
