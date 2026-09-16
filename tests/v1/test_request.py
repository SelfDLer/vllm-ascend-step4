# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus, StreamingUpdate


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert (
        f"{RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR}"
        == "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR"
    )
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def test_streaming_update_reenters_prefill_before_scheduling():
    session = Request(
        request_id="session",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=2),
        pooling_params=None,
        resumable=True,
    )
    session.append_output_token_ids(3)
    session.spec_token_ids = [5, 6]
    session.num_computed_tokens = 3
    session.is_prefill_chunk = False
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ

    update = StreamingUpdate(
        mm_features=None,
        prompt_token_ids=[4],
        max_tokens=2,
        arrival_time=1.0,
        sampling_params=SamplingParams(max_tokens=2),
    )
    scheduler = SimpleNamespace(num_waiting_for_streaming_input=1, log_stats=False)

    Scheduler._update_request_as_session(scheduler, session, update)

    assert session.is_prefill_chunk
    assert session.spec_token_ids == []
    assert session.num_computed_tokens < session.num_prompt_tokens
    assert session.status == RequestStatus.WAITING
