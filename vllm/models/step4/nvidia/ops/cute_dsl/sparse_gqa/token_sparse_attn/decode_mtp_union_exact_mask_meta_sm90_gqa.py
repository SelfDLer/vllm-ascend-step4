# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..indexer_ops.region_sparse_meta_gqa import (
    _launch_grouped_union_logical_bitset_cutedsl,
)


MTP_Q_GROUP = 4
UNION_SPLITS = 8


@triton.jit
def _unpack_decode_packed_regions_kernel(
    packed_ptr,
    counts_ptr,
    logical_ptr,
    phys_ptr,
    stride_packed_q,
    stride_packed_w,
    stride_logical_q,
    stride_logical_w,
    stride_phys_q,
    stride_phys_w,
    rows: tl.constexpr,
    windows: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    valid = (row < rows) & (offsets < windows) & (
        offsets < tl.load(counts_ptr + row)
    )
    packed = tl.load(
        packed_ptr + row * stride_packed_q + offsets * stride_packed_w,
        mask=valid,
        other=-1,
    ).to(tl.int64)
    logical_start = (packed & 0xFFFF_FFFF).to(tl.int32)
    phys_region = ((packed >> 32) & 0xFFFF_FFFF).to(tl.int32)
    logical_start = tl.where(valid & (packed >= 0), logical_start, -1)
    phys_region = tl.where(valid & (packed >= 0), phys_region, -1)
    tl.store(
        logical_ptr + row * stride_logical_q + offsets * stride_logical_w,
        logical_start,
        mask=(row < rows) & (offsets < windows),
    )
    tl.store(
        phys_ptr + row * stride_phys_q + offsets * stride_phys_w,
        phys_region,
        mask=(row < rows) & (offsets < windows),
    )


@triton.jit
def _split_decode_mtp_union_kernel(
    full_union_phys_ptr,
    full_union_logical_ptr,
    full_exact_mask_ptr,
    full_union_counts_ptr,
    seq_lens_ptr,
    split_union_phys_ptr,
    split_union_logical_ptr,
    split_exact_mask_ptr,
    split_union_counts_ptr,
    work_q_global_ptr,
    work_q_input_local_ptr,
    work_q_output_base_ptr,
    work_q_len_ptr,
    causal_limits_ptr,
    full_union_width: tl.constexpr,
    split_union_width: tl.constexpr,
    work_count: tl.constexpr,
    mtp_q_group: tl.constexpr,
    union_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    work = tl.program_id(0)
    request = work // union_splits
    split = work - request * union_splits
    union_count = tl.load(full_union_counts_ptr + request)
    begin = union_count * split // union_splits
    end = union_count * (split + 1) // union_splits
    split_count = end - begin
    offsets = tl.arange(0, BLOCK)
    output_mask = (work < work_count) & (offsets < split_union_width)
    valid = output_mask & (offsets < split_count)
    source = request * full_union_width + begin + offsets
    phys = tl.load(full_union_phys_ptr + source, mask=valid, other=-1)
    logical = tl.load(full_union_logical_ptr + source, mask=valid, other=-1)
    exact_mask = tl.load(full_exact_mask_ptr + source, mask=valid, other=0)
    destination = work * split_union_width + offsets
    tl.store(split_union_phys_ptr + destination, phys, mask=output_mask)
    tl.store(split_union_logical_ptr + destination, logical, mask=output_mask)
    tl.store(split_exact_mask_ptr + destination, exact_mask, mask=output_mask)

    q_input_base = request * mtp_q_group
    tl.store(split_union_counts_ptr + work, split_count)
    tl.store(work_q_global_ptr + work, 0)
    tl.store(work_q_input_local_ptr + work, q_input_base)
    tl.store(
        work_q_output_base_ptr + work,
        q_input_base * union_splits + split,
    )
    tl.store(work_q_len_ptr + work, q_input_base + mtp_q_group)
    slot = offsets
    slot_mask = (work < work_count) & (slot < mtp_q_group)
    seq_len = tl.load(
        seq_lens_ptr + request * mtp_q_group + slot,
        mask=slot_mask,
        other=0,
    )
    tl.store(
        causal_limits_ptr + work * mtp_q_group + slot,
        seq_len,
        mask=slot_mask,
    )


def build_decode_mtp_union_exact_mask_meta_sm90_gqa(
    *,
    region_counts: torch.Tensor,
    region_packed_indices: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    unpacked_logical: torch.Tensor,
    unpacked_phys: torch.Tensor,
    bitset: torch.Tensor,
    full_out_req_idx: torch.Tensor,
    full_work_q_global: torch.Tensor,
    full_work_q_local: torch.Tensor,
    full_work_q_len: torch.Tensor,
    full_union_counts: torch.Tensor,
    full_union_phys: torch.Tensor,
    full_union_logical: torch.Tensor,
    full_causal_limits: torch.Tensor,
    full_exact_mask: torch.Tensor,
    split_union_counts: torch.Tensor,
    split_union_phys: torch.Tensor,
    split_union_logical: torch.Tensor,
    split_exact_mask: torch.Tensor,
    work_q_global: torch.Tensor,
    work_q_input_local: torch.Tensor,
    work_q_output_base: torch.Tensor,
    work_q_len: torch.Tensor,
    causal_limits: torch.Tensor,
    num_region_bins: int,
) -> None:
    rows, windows = (int(v) for v in region_packed_indices.shape)
    if rows % MTP_Q_GROUP != 0:
        raise ValueError(
            f"MTP union metadata requires rows divisible by four, got {rows}"
        )
    requests = rows // MTP_Q_GROUP
    work_count = requests * UNION_SPLITS
    full_union_width = int(full_union_phys.shape[1])
    split_union_width = int(split_union_phys.shape[1])
    if full_union_width < MTP_Q_GROUP * windows:
        raise ValueError(
            "full union capacity must cover every input region: "
            f"capacity={full_union_width}, required={MTP_Q_GROUP * windows}"
        )
    if split_union_width * UNION_SPLITS < full_union_width:
        raise ValueError(
            "split union capacity must cover the full union: "
            f"capacity={split_union_width * UNION_SPLITS}, "
            f"required={full_union_width}"
        )
    expected_shapes = {
        "region_counts": (rows,),
        "region_packed_indices": (rows, windows),
        "kv_seq_lens": (rows,),
        "unpacked_logical": (rows, windows),
        "unpacked_phys": (rows, windows),
        "bitset": (requests, (int(num_region_bins) + 31) // 32),
        "full_out_req_idx": (requests,),
        "full_work_q_global": (requests,),
        "full_work_q_local": (requests,),
        "full_work_q_len": (requests,),
        "full_union_counts": (requests,),
        "full_union_phys": (requests, full_union_width),
        "full_union_logical": (requests, full_union_width),
        "full_causal_limits": (requests, MTP_Q_GROUP),
        "full_exact_mask": (requests, full_union_width),
        "split_union_counts": (work_count,),
        "split_union_phys": (work_count, split_union_width),
        "split_union_logical": (work_count, split_union_width),
        "split_exact_mask": (work_count, split_union_width),
        "work_q_global": (work_count,),
        "work_q_input_local": (work_count,),
        "work_q_output_base": (work_count,),
        "work_q_len": (work_count,),
        "causal_limits": (work_count, MTP_Q_GROUP),
    }
    tensors = locals()
    for name, shape in expected_shapes.items():
        tensor = tensors[name]
        if tuple(int(v) for v in tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if tensor.device != region_packed_indices.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the input CUDA device")
    if region_packed_indices.dtype != torch.int64:
        raise ValueError("region_packed_indices must be int64")
    for name, shape in expected_shapes.items():
        if name == "region_packed_indices":
            continue
        if tensors[name].dtype != torch.int32:
            raise ValueError(f"{name} must be int32")

    unpack_block = triton.next_power_of_2(windows)
    _unpack_decode_packed_regions_kernel[(rows,)](
        region_packed_indices,
        region_counts,
        unpacked_logical,
        unpacked_phys,
        region_packed_indices.stride(0),
        region_packed_indices.stride(1),
        unpacked_logical.stride(0),
        unpacked_logical.stride(1),
        unpacked_phys.stride(0),
        unpacked_phys.stride(1),
        rows=rows,
        windows=windows,
        BLOCK=unpack_block,
        num_warps=4,
    )
    bit_words = (int(num_region_bins) + 31) // 32
    bit_words_width = triton.next_power_of_2(bit_words)
    _launch_grouped_union_logical_bitset_cutedsl(
        logical_sq=unpacked_logical,
        phys_sq=unpacked_phys,
        counts_sq=region_counts,
        bitset=bitset,
        out_req_idx=full_out_req_idx,
        work_q_global=full_work_q_global,
        work_q_local=full_work_q_local,
        work_q_len=full_work_q_len,
        union_phys=full_union_phys,
        union_logical=full_union_logical,
        union_counts=full_union_counts,
        causal_limits=full_causal_limits,
        exact_mask_bits=full_exact_mask,
        total_groups=requests,
        total_q=rows,
        q_global_offset=0,
        q_group=MTP_Q_GROUP,
        topk_windows=windows,
        topk_width=triton.next_power_of_2(windows),
        max_union_windows=full_union_width,
        num_region_bins=int(num_region_bins),
        bit_words=bit_words,
        bit_words_width=bit_words_width,
        emit_causal_limits=True,
        emit_exact_mask=True,
        explicit_work_ranges=False,
    )
    _split_decode_mtp_union_kernel[(work_count,)](
        full_union_phys,
        full_union_logical,
        full_exact_mask,
        full_union_counts,
        kv_seq_lens,
        split_union_phys,
        split_union_logical,
        split_exact_mask,
        split_union_counts,
        work_q_global,
        work_q_input_local,
        work_q_output_base,
        work_q_len,
        causal_limits,
        full_union_width=full_union_width,
        split_union_width=split_union_width,
        work_count=work_count,
        mtp_q_group=MTP_Q_GROUP,
        union_splits=UNION_SPLITS,
        BLOCK=triton.next_power_of_2(split_union_width),
        num_warps=4,
    )


__all__ = ["build_decode_mtp_union_exact_mask_meta_sm90_gqa"]
