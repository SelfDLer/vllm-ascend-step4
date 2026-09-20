# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOTS = (
    REPO_ROOT / "vllm" / "models" / "step5" / "nvidia" / "ops" / "cute_dsl",
    REPO_ROOT / "vllm" / "models" / "step5" / "nvidia" / "ops" / "triton",
    REPO_ROOT / "csrc" / "libtorch_stable" / "step5",
)
LEGACY_TOP_LEVEL_PATHS = (
    REPO_ROOT / "optimus_cutedsl",
    REPO_ROOT / "optimus_triton",
    REPO_ROOT / "optimus_jit_compile_log.py",
)
IGNORED_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".so"}
NON_RUNTIME_PARTS = {
    "benchmark",
    "benchmarks",
    "experiment",
    "experimental",
    "testing",
    "tests",
    "training",
}


def _runtime_files(root: Path):
    return (
        path
        for path in root.rglob("*")
        if path.is_file()
        and not IGNORED_PARTS.intersection(path.parts)
        and path.suffix not in IGNORED_SUFFIXES
    )


def test_step5_kernel_roots_contain_runtime_files_only():
    for root in KERNEL_ROOTS:
        assert root.is_dir(), f"missing Step5 kernel root: {root}"
        for path in _runtime_files(root):
            relative_path = path.relative_to(root)
            assert not NON_RUNTIME_PARTS.intersection(relative_path.parts), path
            assert "_bwd" not in path.name, path
            assert "backward" not in path.name, path
            assert path.name != "optimus_jit_compile_log.py", path


def test_legacy_top_level_kernel_packages_are_not_shipped():
    for path in LEGACY_TOP_LEVEL_PATHS:
        assert not path.exists(), f"legacy top-level kernel path remains: {path}"
