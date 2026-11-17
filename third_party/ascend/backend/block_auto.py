# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""JIT/cache callbacks for the native CPU selector; no device execution."""
import json


def _native(action, contract, **inputs):
    from triton._C.libtriton import ascend
    contract = dict(contract, argument_order=contract.get("argument_order", list(contract.get("arguments", {}))))
    try:
        return json.loads(
            ascend.select_block_candidates(json.dumps(dict(action=action, contract=contract, **inputs),
                                                      allow_nan=False)))
    except RuntimeError as error:
        raise ValueError(str(error)) from error


def candidate_plan(contract, *, probe, axis=-1):
    """Ask the frontend about the axes requested by the native CV planner."""
    requested = _native("plan", contract, axis=axis)["probes"]
    reports = []
    for item in requested:
        try:
            reports.append(dict(item, report=probe(item["axis"], item["factor"])))
        except ValueError:
            continue
    return _native("plan", contract, axis=axis, reports=reports)


def compile_auto_bundle(backend, src, options, compile_variant):
    """Rank TTIR, then compile the best candidate with one F1 fallback attempt."""
    from triton._C.libtriton import ascend, get_cache_invalidating_env_vars, ir
    from . import block_batching as batching
    from . import block_batching_worker as compilation
    from .block_batching_report import _has_dot
    from .block_batching_worker import BatchDecline, CandidateCompiler, _variant_key
    from .compiler import _is_a5_target_arch, make_ttir

    common = {k: v for k, v in options.items() if k not in ("block_auto_contract", "block_batching_grid")}
    baseline_options = dict(common, block_batching_factor=1, block_batching_axis=-1)
    if not options.get("block_auto_contract") or not _is_a5_target_arch(backend.target.arch):
        return compile_variant(baseline_options)
    contract = json.loads(options["block_auto_contract"])
    normalized = backend.parse_options(options)
    refusal = compilation._freeze_refusal(normalized) or compilation._candidate_refusal(backend, src, normalized)
    if refusal:
        return compile_variant(baseline_options)
    positions = compilation._runtime_positions(src)
    if isinstance(positions, BatchDecline):
        return compile_variant(baseline_options)
    scalar_arguments = [contract["arguments"].get(src.fn.arg_names[index], {}).get("value") for index in positions]
    frozen = compilation._freeze_source(backend, src, baseline_options, normalized.block_batching_backend_revision)
    if isinstance(frozen, BatchDecline):
        return compile_variant(baseline_options)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    frontends = {}
    env_vars = get_cache_invalidating_env_vars()

    def frontend(axis, factor):
        values = dict(common, block_batching_factor=factor, block_batching_axis=axis)
        if factor > 1:
            values["block_batching_grid"] = tuple(contract["grid"])
        opts = backend.parse_options(values)
        module = frozen.make_ir(backend.target, opts, backend.get_codegen_implementation(opts),
                                backend.get_module_map(), context)
        cv = _has_dot(module)
        metadata = dict(vars(opts), hash=_variant_key(frozen, backend, opts, env_vars))
        result = make_ttir(module, metadata, opts)
        if isinstance(result, BatchDecline):
            raise ValueError(result.reason)
        report = metadata.get("block_batching", {})
        frontends[axis, factor] = module, report, cv
        return report

    frontend(-1, 1)
    cv = frontends[-1, 1][2]
    axis = options.get("block_batching_axis", -1)
    if cv:
        plan = candidate_plan(contract, axis=axis, probe=frontend)
    else:
        extents = contract["grid"] if axis == -1 else [contract["grid"][axis]]
        specs = [dict(axis=axis, factor=f) for f in (2, 4) if max(extents) >= f]
        plan = dict(candidates=specs, reason="VV_F2_F4_AUTOMATIC_AXIS")
    rows = []
    specs = [dict(axis=-1, factor=1), *plan["candidates"]]
    for spec in specs:
        requested_axis, factor = spec["axis"], spec["factor"]
        try:
            if (requested_axis, factor) not in frontends:
                frontend(requested_axis, factor)
        except ValueError as error:
            rows.append(dict(factor=factor, axis=requested_axis, status="COMPILE_REJECTED", error=str(error)))
            continue
        module, report, _ = frontends[requested_axis, factor]
        selected_axis = report.get("selected_axis", -1)
        config = dict(contract=contract, scalar_arguments=scalar_arguments, factor=factor, axis=selected_axis, cv=cv,
                      preserve_grid=report.get("vv_lowering") == "sequential_iterations_v1")
        row = json.loads(ascend.score_block_ttir(module, json.dumps(config)))
        row.update(cv=cv, merge_kind=spec.get("merge_kind", "unmerged" if factor == 1 else "vv"))
        rows.append(row)
    selection = _native("rank_ttir", contract, candidates=rows)
    selection.pop("factor")
    baseline_key = _variant_key(frozen, backend, backend.parse_options(baseline_options), env_vars)
    baseline = batching.DeferredBaseline(baseline_key, lambda: compile_variant(baseline_options, frozen))
    declines = []
    result = None
    factor = 1
    # Reserve the second backend attempt for F1; no layout retry in auto mode.
    for key in selection["ranked_candidates"][:1]:
        if key == "unmerged":
            break
        row = selection["candidates"][key]
        requested_factor, selected_axis = row["factor"], row["axis"]
        values = dict(common, block_batching_factor=requested_factor, block_batching_axis=selected_axis,
                      block_batching_grid=tuple(contract["grid"]))
        request = batching.BatchRequest(requested_factor, selected_axis)
        candidate = CandidateCompiler(backend, frozen, env_vars).compile(baseline, values, request, positions,
                                                                         defer_fallback=True, allow_layout_retry=False)
        if isinstance(candidate, BatchDecline):
            declines.append(dict(key=key, **candidate.payload()))
            break
        result, factor = candidate, requested_factor
        selection.update(selected_axis=selected_axis, selected_key=key)
        break
    if result is None:
        selection.update(selected_axis=-1, selected_key="unmerged")
        if declines:
            selection["reason"] = "BASELINE_AFTER_CANDIDATE_REJECTION"
        report = dict(compile_status="DISABLED", owner="none", requested_factor=1, applied_factor=1,
                      reason=selection["reason"])
        result = batching.BatchKernel(baseline.materialize(), None, batching.BatchRequest(), report, [])
    result.block_auto_selection = dict(selection, selected_factor=factor, candidate_plan=plan,
                                       compile_rejections=declines)
    return result
