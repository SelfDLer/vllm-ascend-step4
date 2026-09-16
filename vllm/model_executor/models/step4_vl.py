# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Step4 vision-language model."""

from .step3p7 import Step3p7ForConditionalGeneration


class Step4ForConditionalGeneration(Step3p7ForConditionalGeneration):
    """Step4 VLM using the Step3p7 multimodal frontend and Step4 text model."""

    _enable_weights_track_by_default = True
