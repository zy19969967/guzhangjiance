import os
import sys
import tempfile
import types

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_LOADER_ROOT = os.path.join(REPO_ROOT, "data_loader")
if DATA_LOADER_ROOT not in sys.path:
    sys.path.insert(0, DATA_LOADER_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

sys.modules.setdefault("aug", types.SimpleNamespace())
sys.modules.setdefault("data_utils", types.SimpleNamespace())

import data_loader.load as load_module
import data_loader.conditional_load as conditional_load_module
import load_methods


def _dummy_loader(_item_path):
    return np.arange(10, dtype=np.float32).reshape(-1, 1)


def _write_dummy_file(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("dummy")


def _assert_window_starts(windows, starts):
    assert [window.shape[0] for window in windows] == [4] * len(starts)
    assert [int(window[0, 0]) for window in windows] == starts


def test_get_files_keeps_fixed_length_windows_for_any_stride():
    setattr(load_methods, "DUMMY", _dummy_loader)
    with tempfile.TemporaryDirectory() as root:
        class_dir = os.path.join(root, "fault")
        os.makedirs(class_dir)
        _write_dummy_file(os.path.join(class_dir, "sample.txt"))

        labels = {"fault": 7}
        windows, actual_labels = load_module.get_files(
            root,
            "DUMMY",
            ["fault"],
            labels,
            signal_size=4,
            stride=2,
        )
        _assert_window_starts(windows, [0, 2, 4, 6])
        assert actual_labels == [7, 7, 7, 7]

        windows, _ = load_module.get_files(
            root,
            "DUMMY",
            ["fault"],
            labels,
            signal_size=4,
            stride=4,
        )
        _assert_window_starts(windows, [0, 4])

        windows, _ = load_module.get_files(
            root,
            "DUMMY",
            ["fault"],
            labels,
            signal_size=4,
            stride=6,
        )
        _assert_window_starts(windows, [0, 6])


def test_conditional_get_files_keeps_fixed_length_windows_for_any_stride():
    setattr(load_methods, "DUMMY", _dummy_loader)
    with tempfile.TemporaryDirectory() as root:
        class_dir = os.path.join(root, "condition_3", "fault")
        os.makedirs(class_dir)
        _write_dummy_file(os.path.join(class_dir, "sample.txt"))

        labels = {"fault": 7}
        windows, actual_labels = conditional_load_module.get_files(
            root,
            "DUMMY",
            ["fault"],
            labels,
            signal_size=4,
            condition=3,
            stride=2,
        )
        _assert_window_starts(windows, [0, 2, 4, 6])
        assert actual_labels == [7, 7, 7, 7]
