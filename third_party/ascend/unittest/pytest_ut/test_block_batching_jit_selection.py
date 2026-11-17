# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host JIT launch contracts and manual factor preservation; no NPU execution."""
import importlib
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from block_batching_test_utils import assign_npu  # noqa: F401 - CPU-only fixture

BACKEND = Path(__file__).resolve().parents[2] / "backend"


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


PACKAGE = "_block_auto_private_test"


class Tensor:
    dtype = "torch.float32"
    shape = (256, 512)

    def stride(self):
        return (512, 1)

    def storage_offset(self):
        return 0


class TestHostIntegration(unittest.TestCase):

    def setUp(self):
        self.calls = []
        self.config = module(PACKAGE + ".reuse.config", enabled=lambda: False)
        self.patcher = patch.dict(
            sys.modules, {
                PACKAGE: module(PACKAGE, __path__=[str(BACKEND)]),
                PACKAGE + ".reuse": module(PACKAGE + ".reuse", __path__=[]),
                PACKAGE + ".reuse.config": self.config,
            })
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.batching = importlib.import_module(PACKAGE + ".block_batching")
        self.dispatch = self.batching
        capture = patch.object(self.batching, "_capture_state", return_value=False)
        self.capture = capture.start()
        self.addCleanup(capture.stop)
        self.backend = SimpleNamespace(target=SimpleNamespace(arch="Ascend950"), prepare_block_compile_options=dict)
        self.fn = SimpleNamespace(arg_names=["x", "size", "BLOCK"], device_caches={0: [None] * 4 + [self.bind]})
        self.kwargs = dict(block_batching_factor="auto", BLOCK=512)
        self.result = object()

    def bind(self, *args, **kwargs):
        self.calls.append("bind")
        return dict(x=args[0], size=args[1], BLOCK=kwargs["BLOCK"]), [("*fp32", None), ("i32", None),
                                                                      ("constexpr", 512)], kwargs

    def tail(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result

    def run_handler(self, grid=(256, ), warmup=False):
        return self.batching.get_launch_handler().run(self.fn, (Tensor(), 131072), self.kwargs, grid, warmup, 0, 42,
                                                      self.backend, self.tail)

    def test_auto_contract_resolves_grid_once_and_excludes_constexpr(self):

        def grid(bound):
            self.calls.append("grid")
            return (bound["size"] // bound["BLOCK"], )

        self.assertIs(self.run_handler(grid), self.result)
        self.assertEqual(self.calls[:2], ["bind", "grid"])
        self.assertEqual(len(self.calls), 3)
        args, kwargs = self.calls[-1]
        self.assertEqual(args[2], (256, ))
        contract = json.loads(args[1]["block_auto_contract"])
        self.assertEqual(contract["argument_order"], ["x", "size"])
        self.assertEqual(contract["grid"], [256, 1, 1])
        self.assertEqual(contract["arguments"]["x"]["shape"], [256, 512])
        self.assertNotIn("block_auto_contract", self.kwargs)

    def test_omitted_factor_matches_explicit_auto(self):
        explicit, _ = self.dispatch.prepare_launch(self.fn, (Tensor(), 131072), self.kwargs, (256, ), False, 0,
                                                   self.backend)
        self.kwargs.pop("block_batching_factor")
        self.calls.clear()
        self.run_handler()
        prepared = self.calls[-1][0][1]
        self.assertEqual(prepared, explicit)
        self.assertEqual(list(prepared), list(explicit))
        self.assertEqual(prepared["block_batching_factor"], "auto")
        self.assertNotIn("block_batching_factor", self.kwargs)

    def test_incomplete_tensor_metadata_does_not_break_default_launch(self):
        wrapper = SimpleNamespace(dtype="fp32", shape=(256, 512), stride=lambda: (512, 1))
        self.assertEqual(self.dispatch._argument(wrapper), dict(kind="unsupported"))

    def test_float_launch_arguments_keep_finite_values_only(self):
        for value in (0.75, float("inf"), float("nan")):
            with self.subTest(value=value):
                options, _ = self.dispatch.prepare_launch(self.fn, (Tensor(), value), self.kwargs, (256, ), False, 0,
                                                          self.backend)
                argument = json.loads(options["block_auto_contract"])["arguments"]["size"]
                expected = dict(kind="scalar", value=value) if value == 0.75 else dict(kind="unsupported")
                self.assertEqual(argument, expected)

    def test_explicit_factors_keep_manual_dispatch(self):
        for factor in (1, 2, 4):
            context = ("Ascend950", 24 // factor, (True, 24 // factor, 48 // factor), None)
            self.backend.prepare_block_compile_options = Mock(
                side_effect=lambda values: dict(values, block_compile_context=context))
            self.kwargs = dict(BLOCK=512, block_batching_axis=0)
            self.kwargs["block_batching_factor"] = factor
            self.calls.clear()
            self.assertIs(self.run_handler(), self.result)
            self.assertEqual(len(self.calls), 1)
            if factor == 1:
                self.assertIs(self.calls[0][0][1], self.kwargs)
                self.backend.prepare_block_compile_options.assert_not_called()
            else:
                self.assertEqual(self.calls[0][0][1], dict(self.kwargs, block_compile_context=context))
                self.backend.prepare_block_compile_options.assert_called_once_with(self.kwargs)

    def test_warmup_uses_the_same_contract_as_launch(self):
        self.kwargs.pop("block_batching_factor")
        self.run_handler(warmup=True)
        warmed = self.calls[-1][0][1]
        self.run_handler()
        self.assertEqual(warmed, self.calls[-1][0][1])
        self.assertEqual(warmed["block_batching_factor"], "auto")
        incomplete = SimpleNamespace(dtype="fp32", shape=(1, ), stride=lambda: (1, ))
        options, _ = self.dispatch.prepare_launch(self.fn, (incomplete, 131072), self.kwargs, (256, ), True, 0,
                                                  self.backend)
        self.assertEqual(options["block_batching_factor"], 1)
        self.assertNotIn("block_auto_contract", options)

    def test_capture_uses_baseline_without_binding_or_grid_evaluation(self):
        self.kwargs.pop("block_batching_factor")

        def unavailable(_):
            self.fail("grid must stay lazy")

        for warmup, capture in ((True, True), (False, True), (False, None)):
            self.calls.clear()
            self.capture.return_value = capture
            self.run_handler(unavailable, warmup)
            self.assertEqual(len(self.calls), 1)
            values, _ = self.calls[0]
            self.assertEqual(values[1]["block_batching_factor"], 1)
            self.assertIs(values[2], unavailable)

    def test_non_a5_and_invalid_grid_keep_baseline(self):
        self.kwargs.pop("block_batching_factor")
        for target, grid in (("Ascend910B", (256, )), ("Ascend950", (0, ))):
            self.backend.target.arch = target
            self.calls.clear()
            self.run_handler(grid)
            self.assertEqual(self.calls[-1][0][1]["block_batching_factor"], 1)

    def test_a3_enters_auto_without_a5_empirical_override(self):
        self.backend.target.arch = "Ascend910_9391"
        self.run_handler()
        options = self.calls[-1][0][1]
        self.assertEqual(options["block_batching_factor"], "auto")
        self.assertFalse(json.loads(options["block_auto_contract"])["cv_empirical_short_batch"])

    def test_single_program_keeps_normal_cache_without_contract(self):
        for grid, axis in (((1, ), -1), ((1, 1, 1), -1), ((8, 1), 1)):
            self.kwargs["block_batching_axis"] = axis
            self.calls.clear()
            self.run_handler(grid)
            options = self.calls[-1][0][1]
            self.assertEqual(options["block_batching_factor"], 1)
            self.assertNotIn("block_auto_contract", options)

    def test_other_frontends_keep_their_original_compilation_path(self):
        self.fn.ASTSource = SimpleNamespace(frontend_kind="tla")
        self.kwargs.pop("block_batching_factor")
        self.run_handler()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][0][1]["block_batching_factor"], 1)

    def test_shape_and_scalar_values_are_part_of_jit_option_identity(self):
        first, _ = self.dispatch.prepare_launch(self.fn, (Tensor(), 131072), self.kwargs, (256, ), False, 0,
                                                self.backend)
        second, _ = self.dispatch.prepare_launch(self.fn, (Tensor(), 65536), self.kwargs, (128, ), False, 0,
                                                 self.backend)
        self.assertNotEqual(first["block_auto_contract"], second["block_auto_contract"])

    def test_reuse_keeps_original_grid_and_unfragmented_options(self):
        self.config.enabled = lambda: True
        grid = Mock(side_effect=AssertionError("reuse must receive the unevaluated grid"))

        def reuse(fn, args, kwargs, grid, warmup, device, stream, backend, tail):
            self.assertIs(fn, self.fn)
            self.assertNotIn("block_auto_contract", kwargs)
            self.assertEqual(kwargs["block_batching_factor"], 1)
            return tail(args, kwargs, grid, warmup, device, stream)

        with patch.dict(sys.modules,
                        {PACKAGE + ".reuse.bridge": module("bridge", get_handler=lambda: SimpleNamespace(run=reuse))}):
            self.assertIs(self.run_handler(grid), self.result)
        grid.assert_not_called()
        self.assertNotIn("bind", self.calls)
        self.assertEqual(len([item for item in self.calls if isinstance(item, tuple)]), 1)

    def test_equivalent_auto_decisions_share_device_handle_and_identity(self):
        cache = {}

        class Kernel:
            hash = "same-artifact"
            block_batching = {"reason": "DISABLED"}
            block_auto_selection = {"selected_factor": 1, "reason": "GRID_BELOW_CORE_FLOOR"}

            def __init__(self):
                self._init_handles = Mock()

        first, second = Kernel(), Kernel()
        with patch.object(self.batching, "BatchKernel", Kernel):
            self.assertIs(self.dispatch.initialize_auto_kernel(cache, first), first)
            self.assertIs(self.dispatch.initialize_auto_kernel(cache, second), first)
            second._init_handles.assert_not_called()
            # A different device/native cache cannot reuse the previous handle.
            self.assertIs(self.dispatch.initialize_auto_kernel({}, second), second)
            changed = Kernel()
            changed.block_auto_selection = {"selected_factor": 1, "reason": "MODEL_INCOMPLETE"}
            self.assertIs(self.dispatch.initialize_auto_kernel(cache, changed), changed)
            cache.clear()
            self.assertIs(self.dispatch.initialize_auto_kernel(cache, second), second)

    def test_launch_exception_is_not_replayed(self):

        def fail(*args, **kwargs):
            self.calls.append("launch")
            raise RuntimeError("launch error")

        self.tail = fail
        with self.assertRaisesRegex(RuntimeError, "launch error"):
            self.run_handler()
        self.assertEqual(self.calls.count("launch"), 1)
