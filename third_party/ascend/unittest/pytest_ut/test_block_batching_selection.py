# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Frontend scoring, bounded compilation and deferred guarded fallback; CPU-only."""
from block_batching_test_utils import assign_npu  # noqa: F401 - CPU-only fixture
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from triton._C.libtriton import ascend, ir
from triton.backends.ascend import block_batching as batching
from triton.backends.ascend import block_batching_guard
from triton.backends.ascend.compiler import AscendBackend
from triton.backends.compiler import GPUTarget


class TestDeferredFallback(unittest.TestCase):

    def setUp(self):
        self.events = []

        def kernel(role):
            return SimpleNamespace(hash=role, metadata=SimpleNamespace(_asdict=lambda: {}), function=role,
                                   packed_metadata=role, launch_metadata=lambda *args: None, _init_handles=Mock(),
                                   _run=True, run=lambda *args: self.events.append((args[4], args[5])))

        self.original = kernel("f1")
        self.candidate = kernel("f2")
        self.compile_f1 = Mock(return_value=self.original)
        self.deferred = batching.DeferredBaseline("f1", self.compile_f1)
        self.guard = Mock(return_value=(True, "PROVEN_AT_RUNTIME"))
        with patch.object(block_batching_guard, "RuntimeGuard", return_value=self.guard):
            self.bundle = batching.BatchKernel(
                self.deferred, self.candidate, batching.BatchRequest(2, 0),
                dict(cv_mode="independent_batch_v1", selected_axis=0, lane_width=2, guard={}, owner="block_batching"),
                [0], publish_manifest=False)
        self.bundle.block_auto_selection = dict(selected_factor=2)

    def dispatch(self, *, capture=False):
        with patch.object(batching, "_capture_state", return_value=capture), \
             patch.object(batching, "_core_floor", return_value=56):
            self.bundle._dispatch(4, 1, 1, 0, None, None, None, None, None, 1234)

    def test_successful_candidate_does_not_compile_or_load_f1(self):
        with patch.object(batching, "_capture_state", return_value=False):
            self.bundle._init_handles()
        self.dispatch()
        self.compile_f1.assert_not_called()
        self.original._init_handles.assert_not_called()
        self.assertEqual(self.events, [("f2", "f2")])

    def test_alias_fallback_materializes_once_and_forwards_complete_f1(self):
        self.guard.return_value = False, "WRITE_ALIAS"
        self.dispatch()
        self.dispatch()
        self.compile_f1.assert_called_once()
        self.assertEqual(self.events, [("f1", "f1"), ("f1", "f1")])
        self.assertEqual(self.bundle.last_launch["applied_factor"], 1)

    def test_capture_does_not_trigger_lazy_compilation(self):
        with self.assertRaisesRegex(RuntimeError, "before graph capture"):
            self.dispatch(capture=True)
        self.compile_f1.assert_not_called()
        self.assertFalse(self.events)

    def test_capture_can_reuse_an_already_prepared_baseline(self):
        self.deferred.materialize()
        self.dispatch(capture=True)
        self.compile_f1.assert_called_once()
        self.assertEqual(self.events, [("f1", "f1")])

    def test_compilation_error_is_not_silently_swallowed(self):
        self.compile_f1.side_effect = RuntimeError("test compile error")
        with self.assertRaisesRegex(RuntimeError, "test compile error"):
            self.deferred.materialize()
        self.assertFalse(self.events)

    def test_candidate_load_resource_failure_uses_complete_f1(self):
        from triton.runtime.errors import OutOfResources
        self.candidate._init_handles.side_effect = OutOfResources(4096, 1024, "shared memory")
        with patch.object(batching, "_capture_state", return_value=False):
            self.bundle._init_handles()
        self.dispatch()
        self.compile_f1.assert_called_once()
        self.assertIsNone(self.bundle.candidate)
        self.assertEqual(self.events, [("f1", "f1")])


class TestTtirAnalysis(unittest.TestCase):

    def score(self, text, scalar_arguments=None, **options):
        context = ir.context()
        ir.load_dialects(context)
        AscendBackend(GPUTarget("npu", "Ascend950", 32)).load_dialects(context)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "kernel.mlir"
            source.write_text(text)
            module = ir.parse_mlir_module(str(source), context)
        request = dict(contract=dict(grid=[256, 1, 1]), factor=1, axis=-1, cv=False)
        request.update(options)
        if scalar_arguments is not None:
            request["scalar_arguments"] = scalar_arguments
        original = str(module)
        result = json.loads(ascend.score_block_ttir(module, json.dumps(request)))
        self.assertEqual(str(module), original)
        return result

    def test_dynamic_loop_declines_instead_of_assuming_one_iteration(self):
        result = self.score('''module {
          tt.func public @kernel(%n: index, %out: !tt.ptr<i32>) {
            %zero = arith.constant 0 : index
            %one = arith.constant 1 : index
            scf.for %i = %zero to %n step %one {
              %value = arith.addi %i, %one : index
              %stored = arith.index_cast %value : index to i32
              tt.store %out, %stored : !tt.ptr<i32>
              scf.yield
            }
            tt.return
          }
        }''')
        self.assertEqual(result["status"], "TTIR_MODEL_UNSUPPORTED")
        self.assertIn("dynamic loop", result["error"])

    def test_launch_scalar_specializes_loop_without_changing_executable_ir(self):
        text = '''module {
          tt.func public @kernel(%n: index, %out: !tt.ptr<i32>) {
            %zero = arith.constant 0 : index
            %one = arith.constant 1 : index
            scf.for %i = %zero to %n step %one {
              %stored = arith.index_cast %i : index to i32
              tt.store %out, %stored : !tt.ptr<i32>
            }
            tt.return
          }
        }'''
        result = self.score(text, [7, None])
        self.assertEqual(result["status"], "TTIR_COST_ESTIMATED")
        self.assertEqual(result["memory_requests"], 7)
        self.assertFalse(result["loop_trip_upper_bound"])

    def test_persistent_loop_uses_grid_range_for_worst_core(self):
        result = self.score('''module {
          tt.func public @kernel(%out: !tt.ptr<i32>) {
            %pid = tt.get_program_id x : i32
            %grid = tt.get_num_programs x : i32
            %upper = arith.constant 1000 : i32
            scf.for %i = %pid to %upper step %grid : i32 {
              tt.store %out, %i : !tt.ptr<i32>
            }
            tt.return
          }
        }''')
        self.assertEqual(result["status"], "TTIR_COST_ESTIMATED")
        self.assertEqual(result["memory_requests"], 4)
        self.assertTrue(result["loop_trip_upper_bound"])

    def test_frontend_liveness_never_claims_final_ub_validation(self):
        result = self.score('''module {
          tt.func public @kernel(%x: tensor<128xf32>) {
            %v = arith.mulf %x, %x : tensor<128xf32>
            tt.return
          }
        }''')
        self.assertEqual(result["status"], "TTIR_COST_ESTIMATED")
        self.assertFalse(result["resource_checked"])

    def test_equivalent_duplicate_expressions_have_the_same_cost(self):
        template = '''module {
          tt.func public @kernel(%x: tensor<128xf32>, %out: tensor<128x!tt.ptr<f32>>) {
            %a = arith.mulf %x, %x : tensor<128xf32>
            DUPLICATE
            %sum = arith.addf %a, RHS : tensor<128xf32>
            tt.store %out, %sum : tensor<128x!tt.ptr<f32>>
            tt.return
          }
        }'''
        shared = self.score(template.replace("DUPLICATE", "").replace("RHS", "%a"))
        duplicate = self.score(
            template.replace("DUPLICATE", "%b = arith.mulf %x, %x : tensor<128xf32>").replace("RHS", "%b"))
        self.assertEqual(shared["status"], "TTIR_COST_ESTIMATED")
        self.assertEqual(shared["score"], duplicate["score"])
        self.assertEqual(shared["estimated_live_bytes"], duplicate["estimated_live_bytes"])

    def test_scan_cost_accounts_for_retaining_all_prefix_values(self):
        reduction = '''module {
          tt.func public @kernel(%x: tensor<512xf32>, %out: !tt.ptr<f32>) {
            %v = "tt.reduce"(%x) ({
            ^bb0(%a: f32, %b: f32):
              %c = arith.addf %a, %b : f32
              tt.reduce.return %c : f32
            }) {axis = 0 : i32} : (tensor<512xf32>) -> f32
            tt.store %out, %v : !tt.ptr<f32>
            tt.return
          }
        }'''
        scan = '''module {
          tt.func public @kernel(%x: tensor<512xf32>, %out: tensor<512x!tt.ptr<f32>>) {
            %v = "tt.scan"(%x) ({
            ^bb0(%a: f32, %b: f32):
              %c = arith.addf %a, %b : f32
              tt.scan.return %c : f32
            }) {axis = 0 : i32, reverse = false} : (tensor<512xf32>) -> tensor<512xf32>
            tt.store %out, %v : tensor<512x!tt.ptr<f32>>
            tt.return
          }
        }'''
        self.assertGreater(self.score(scan)["vector_work"], self.score(reduction)["vector_work"])

    def test_small_dma_batches_do_not_charge_bytes_for_idle_tail_cores(self):

        def candidate(factor):
            size = 512 * factor
            text = '''module {
              tt.func public @kernel(%ptr: !tt.ptr<f32>) {
                %r = tt.make_range {start = 0 : i32, end = SIZE : i32} : tensor<SIZExi32>
                %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<SIZEx!tt.ptr<f32>>
                %q = tt.addptr %p, %r : tensor<SIZEx!tt.ptr<f32>>, tensor<SIZExi32>
                %x = tt.load %q : tensor<SIZEx!tt.ptr<f32>>
                %y = arith.mulf %x, %x : tensor<SIZExf32>
                tt.store %q, %y : tensor<SIZEx!tt.ptr<f32>>
                tt.return
              }
            }'''.replace("SIZE", str(size))
            return self.score(text, factor=factor, axis=0)

        original, merged = candidate(1), candidate(4)
        self.assertEqual(original["status"], "TTIR_COST_ESTIMATED")
        self.assertEqual(merged["status"], "TTIR_COST_ESTIMATED")
        self.assertEqual(original["blocks"] * original["bytes_per_program"],
                         merged["blocks"] * merged["bytes_per_program"])
        self.assertLess(merged["score"], original["score"])

    def test_batched_scans_retain_serial_helper_cost_per_row(self):

        def candidate(factor):
            shape = "512" if factor == 1 else f"{factor}x512"
            axis = 0 if factor == 1 else 1
            text = '''module {
              tt.func public @kernel(%p: tensor<SHAPEx!tt.ptr<f32>>) {
                %x = tt.load %p : tensor<SHAPEx!tt.ptr<f32>>
                %y = "tt.scan"(%x) ({
                ^bb0(%a: f32, %b: f32):
                  %c = arith.addf %a, %b : f32
                  tt.scan.return %c : f32
                }) {axis = AXIS : i32, reverse = false} :
                    (tensor<SHAPExf32>) -> tensor<SHAPExf32>
                tt.store %p, %y : tensor<SHAPEx!tt.ptr<f32>>
                tt.return
              }
            }'''.replace("SHAPE", shape).replace("AXIS", str(axis))
            return self.score(text, factor=factor, axis=0)

        original = candidate(1)
        self.assertEqual(original["status"], "TTIR_COST_ESTIMATED")
        for factor in (2, 4):
            merged = candidate(factor)
            self.assertEqual(merged["status"], "TTIR_COST_ESTIMATED")
            self.assertGreater(merged["score"], original["score"])

    def test_small_predicted_gain_remains_eligible(self):
        request = dict(
            action="rank_ttir", contract=dict(grid=[256, 1, 1]), candidates=[
                dict(factor=1, axis=-1, status="TTIR_COST_ESTIMATED", score=100, cores=56, blocks=256),
                dict(factor=4, axis=0, status="TTIR_COST_ESTIMATED", score=99.5, cores=56, blocks=64)
            ])
        result = json.loads(ascend.select_block_candidates(json.dumps(request)))
        self.assertEqual(result["factor"], 4)

    def test_missing_baseline_cost_forces_f1(self):
        request = dict(
            action="rank_ttir", contract=dict(grid=[256, 1, 1]), candidates=[
                dict(factor=1, axis=-1, status="TTIR_MODEL_UNSUPPORTED"),
                dict(factor=2, axis=0, status="TTIR_COST_ESTIMATED", score=1, cores=56, blocks=128)
            ])
        result = json.loads(ascend.select_block_candidates(json.dumps(request)))
        self.assertEqual(result["factor"], 1)
        self.assertFalse(result["model_complete"])


class TestRankedRecovery(unittest.TestCase):

    def test_auto_worker_does_not_retry_a_rejected_layout(self):
        from triton.backends.ascend import block_batching_worker as worker
        compiler = worker.CandidateCompiler(Mock(), Mock(), {})
        decline = dict(reason="LAYOUT_TARGET_RESOURCE_REJECTED", stage="npubin", diagnostic="UB capacity exceeded",
                       factor=4, axis=0)
        with patch.object(compiler, "_variant", return_value=(decline, None)) as attempt:
            result = compiler.compile(Mock(), dict(block_batching_layout="auto"), batching.BatchRequest(4, 0), [],
                                      defer_fallback=True, allow_layout_retry=False)
        self.assertIsInstance(result, worker.BatchDecline)
        attempt.assert_called_once()

    def test_cv_planner_keeps_two_spatial_candidates(self):
        from triton.backends.ascend.block_auto import candidate_plan

        def probe(axis, factor):
            return dict(compile_status="REWRITTEN", selected_axis=axis, requested_factor=factor,
                        cv_mode="spatial_concat_v1", cv_lowering="spatial_m_v1" if axis == 0 else "spatial_n_v1")

        plan = candidate_plan(dict(grid=[8, 8, 1]), probe=probe)
        self.assertEqual([(row["axis"], row["factor"]) for row in plan["candidates"]], [(0, 4), (1, 4)])

    def run_selection(self, declines=(), internal_error=False):
        from triton.backends.ascend import block_auto as dispatch
        from triton.backends.ascend import block_batching_worker as worker
        from triton.backends.ascend import compiler, block_batching_report
        backend = Mock(target=GPUTarget("npu", "Ascend950", 32))
        backend.parse_options.side_effect = lambda options: SimpleNamespace(block_batching_backend_revision="test", **
                                                                            options)
        source = SimpleNamespace(fn=SimpleNamespace(arg_names=["X"]))
        baseline = SimpleNamespace(hash="f1")
        compile_f1 = Mock(return_value=baseline)
        attempts = []

        def frontend(module, metadata, options):
            metadata["block_batching"] = dict(selected_axis=0)
            return module

        def score(module, request):
            config = json.loads(request)
            factor = config["factor"]
            return json.dumps(
                dict(status="TTIR_COST_ESTIMATED", factor=factor, axis=0, blocks=448 // factor, cores=56,
                     score=5 - factor))

        def compile_candidate(baseline, options, request, positions, *, defer_fallback=False, allow_layout_retry=True):
            self.assertTrue(defer_fallback)
            self.assertFalse(allow_layout_retry)
            attempts.append(request.factor)
            if internal_error:
                raise RuntimeError("unexpected internal failure")
            if request.factor in declines:
                return worker.BatchDecline("LAYOUT_TARGET_RESOURCE_REJECTED", stage="npubin",
                                           diagnostic="UB capacity exceeded", factor=request.factor, axis=request.axis)
            return SimpleNamespace()

        contract = dict(grid=[448, 1, 1], arguments=dict(X=dict(kind="tensor")))
        options = dict(block_auto_contract=json.dumps(contract), block_batching_factor="auto")
        with ExitStack() as stack:
            for name, value in [("_freeze_refusal", None), ("_candidate_refusal", None), ("_runtime_positions", [0]),
                                ("_freeze_source", Mock())]:
                stack.enter_context(patch.object(worker, name, return_value=value))
            stack.enter_context(patch.object(batching, "BatchKernel", return_value=SimpleNamespace()))
            stack.enter_context(patch.object(block_batching_report, "_has_dot", return_value=False))
            stack.enter_context(patch.object(worker, "_variant_key", return_value="f1"))
            stack.enter_context(patch.object(worker.CandidateCompiler, "compile", side_effect=compile_candidate))
            stack.enter_context(patch.object(compiler, "make_ttir", side_effect=frontend))
            stack.enter_context(patch.object(ascend, "score_block_ttir", side_effect=score))
            result = dispatch.compile_auto_bundle(backend, source, options, compile_f1)
        return result.block_auto_selection, attempts, compile_f1

    def test_rejected_f4_reserves_second_attempt_for_f1(self):
        selection, attempts, compile_f1 = self.run_selection(declines=(4, ))
        self.assertEqual(attempts, [4])
        self.assertEqual(selection["selected_factor"], 1)
        self.assertEqual(selection["reason"], "BASELINE_AFTER_CANDIDATE_REJECTION")
        compile_f1.assert_called_once()

    def test_all_merged_candidates_rejected_compile_f1_once(self):
        selection, attempts, compile_f1 = self.run_selection(declines=(4, 2))
        self.assertEqual(attempts, [4])
        self.assertEqual(selection["selected_factor"], 1)
        self.assertEqual(selection["reason"], "BASELINE_AFTER_CANDIDATE_REJECTION")
        compile_f1.assert_called_once()

    def test_internal_errors_do_not_masquerade_as_resource_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected internal failure"):
            self.run_selection(internal_error=True)


if __name__ == "__main__":
    unittest.main()
