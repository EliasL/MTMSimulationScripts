from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from MTMath.reduction import lagrange_reduction
else:
    from .reduction import lagrange_reduction


def _serialize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def _coerce_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    array_keys = {"F", "C", "C_", "C_R", "dN_dX", "dN_dx"}
    out: dict[str, Any] = {}
    for key, value in inputs.items():
        if key in array_keys:
            out[key] = np.asarray(value, dtype=float)
        else:
            out[key] = value
    return out


def _compare_numeric(result: Any, expected: Any, rtol: float, atol: float) -> bool:
    res = np.asarray(result, dtype=float)
    exp = np.asarray(expected, dtype=float)
    if res.shape != exp.shape:
        return False
    return np.allclose(res, exp, rtol=rtol, atol=atol, equal_nan=True)


def _default_answers_path(
    out_dir: str | Path,
    date_str: str | None = None,
    filename: str | None = None,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    date_str = date_str or date.today().isoformat()
    if filename is None:
        filename = f"energyFunctionAnswers_{date_str}.json"
    return out_path / filename


def _write_answers_json(answers: dict[str, Any], path: Path) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(answers, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


def load_answers(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def generate_answer_dict(
    model_cls,
    beta: float = -1 / 4,
    K: float = 4,
    noise: float = 1,
    loops: int = 1000,
    area: float = 0.5,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    out_dir: str | Path = "energyFunctionTestAnswers",
    date_str: str | None = None,
    filename: str | None = None,
    save_json: bool = True,
):
    """
    Generate a regression answers dict for a given energy model class.
    By default it saves a JSON file for later testing.
    Returns (answers_dict, saved_path).
    """
    dN_dX = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    dN_dx = np.array([[-0.8, -1.2], [0.9, 0.1], [-0.1, 1.1]], dtype=float)

    test_cases = [
        ("simple_shear_0p15", np.array([[1.0, 0.15], [0.0, 1.0]], dtype=float)),
        ("simple_shear_0p801", np.array([[1.0, 0.801], [0.0, 1.0]], dtype=float)),
        (
            "general_deformation",
            np.array([[1.2, 0.1], [0.05, 0.9]], dtype=float),
        ),
    ]

    cases = []
    for name, F in test_cases:
        C = F.T @ F
        C_R, _ = lagrange_reduction(C.copy(), loops=loops)

        inputs = {
            "energy_from_F": {
                "F": F.tolist(),
                "beta": beta,
                "K": K,
                "noise": noise,
                "zeroReference": True,
                "accuracy": 1,
                "loops": loops,
            },
            "energy_from_reduced_C": {
                "C_": C_R.tolist(),
                "beta": beta,
                "K": K,
                "noise": noise,
                "zeroReference": True,
            },
            "energy_from_reduced_C_components": {
                "C11": float(C_R[0, 0]),
                "C22": float(C_R[1, 1]),
                "C12": float(C_R[0, 1]),
                "beta": beta,
                "K": K,
                "noise": noise,
                "zeroReference": True,
            },
            "sigma_from_C_R": {
                "C_R": C_R.tolist(),
                "beta": beta,
                "K": K,
                "noise": noise,
            },
            "S_from_C": {
                "C": C.tolist(),
                "beta": beta,
                "K": K,
                "noise": noise,
            },
            "S_from_F": {
                "F": F.tolist(),
                "beta": beta,
                "K": K,
                "noise": noise,
            },
            "P_from_F": {
                "F": F.tolist(),
                "beta": beta,
                "K": K,
                "noise": noise,
            },
            "cauchy_from_F": {
                "F": F.tolist(),
                "beta": beta,
                "K": K,
                "noise": noise,
            },
            "lagrangian_forces_from_F": {
                "F": F.tolist(),
                "dN_dX": dN_dX.tolist(),
                "beta": beta,
                "K": K,
                "noise": noise,
                "area": area,
            },
            "eulerian_forces_from_F": {
                "F": F.tolist(),
                "dN_dx": dN_dx.tolist(),
                "beta": beta,
                "K": K,
                "noise": noise,
                "area": area,
            },
        }

        expected = {}
        for fn_name, fn_inputs in inputs.items():
            call_inputs = _coerce_inputs(fn_inputs)
            result = getattr(model_cls, fn_name)(**call_inputs)
            expected[fn_name] = _serialize_value(result)

        cases.append({"name": name, "inputs": inputs, "expected": expected})

    answers = {
        "meta": {
            "model": model_cls.__name__,
            "version": 1,
            "rtol": rtol,
            "atol": atol,
        },
        "cases": cases,
    }

    saved_path = None
    if save_json:
        saved_path = _default_answers_path(
            out_dir, date_str=date_str, filename=filename
        )
        _write_answers_json(answers, saved_path)

    return answers, saved_path


def test_against_answers(
    model_cls,
    answers: dict[str, Any] | str | Path,
    rtol: float | None = None,
    atol: float | None = None,
    verbose: bool = True,
):
    """
    Test a model against a saved answers dict.
    Returns (ok, failures), where failures is a list of failure records.
    """
    if isinstance(answers, (str, Path)):
        answers = load_answers(answers)

    meta = answers.get("meta", {})
    rtol = meta.get("rtol", 1e-7) if rtol is None else rtol
    atol = meta.get("atol", 1e-9) if atol is None else atol

    ok_all = True
    failures = []

    cases = answers.get("cases", [])
    for case in cases:
        case_name = case.get("name", "case")
        inputs = case.get("inputs", {})
        expected = case.get("expected", {})

        for fn_name, exp in expected.items():
            if fn_name not in inputs:
                ok_all = False
                failures.append((case_name, fn_name, "missing_inputs"))
                if verbose:
                    print(f"[FAIL] {case_name}::{fn_name} missing inputs")
                continue
            if not hasattr(model_cls, fn_name):
                ok_all = False
                failures.append((case_name, fn_name, "missing_function"))
                if verbose:
                    print(
                        f"[FAIL] {case_name}::{fn_name} missing function on {model_cls.__name__}"
                    )
                continue

            try:
                call_inputs = _coerce_inputs(inputs[fn_name])
                result = getattr(model_cls, fn_name)(**call_inputs)
                ok = _compare_numeric(result, exp, rtol=rtol, atol=atol)
            except Exception as exc:
                ok = False
                failures.append((case_name, fn_name, f"exception: {exc}"))
                if verbose:
                    print(f"[FAIL] {case_name}::{fn_name} raised {exc}")
                continue

            if not ok:
                ok_all = False
                failures.append((case_name, fn_name, "mismatch"))
                if verbose:
                    res_arr = np.asarray(result, dtype=float)
                    exp_arr = np.asarray(exp, dtype=float)
                    if res_arr.shape == exp_arr.shape:
                        max_err = np.max(np.abs(res_arr - exp_arr))
                    else:
                        max_err = float("nan")
                    print(
                        f"[FAIL] {case_name}::{fn_name} "
                        f"max_err={max_err:.6g} rtol={rtol} atol={atol}"
                    )
            elif verbose:
                print(f"[PASS] {case_name}::{fn_name}")

    if verbose:
        print("Summary:", "PASS" if ok_all else "FAIL", f"({len(failures)} failures)")

    return ok_all, failures


if __name__ == "__main__":
    if __package__ in (None, ""):
        from MTMath.energyFunction import ContiEnergy
    else:
        from .energyFunction import ContiEnergy

    # answers, saved_path = generate_answer_dict(ContiEnergy)
    # print("Saved answers to:", saved_path)
    saved_path = "energyFunctionTestAnswers/energyFunctionAnswers_2026-02-17.json"
    if saved_path is None:
        raise RuntimeError("Expected answers to be saved, but no path was returned.")

    test_against_answers(ContiEnergy, saved_path)
