"""Validate JSON fixtures against the Python reference implementation."""
import json
import sys
sys.path.insert(0, ".")

def validate_infer_prefix():
    from dspy.signatures.signature import infer_prefix
    with open("spec/fixtures/infer_prefix.json") as f:
        fixtures = json.load(f)
    failures = []
    for case in fixtures["cases"]:
        result = infer_prefix(case["input"])
        if result != case["expected"]:
            failures.append((case["input"], case["expected"], result))
    return "infer_prefix", len(fixtures["cases"]), failures

def validate_signature_parse():
    from dspy.signatures.signature import _parse_signature
    with open("spec/fixtures/signature_parse.json") as f:
        fixtures = json.load(f)
    failures = []
    for case in fixtures["cases"]:
        if "expected_error" in case:
            try:
                _parse_signature(case["input"])
                failures.append((case["id"], "expected error", "no error"))
            except (ValueError, Exception):
                pass
        else:
            try:
                result = _parse_signature(case["input"])
                for inp in case["expected"]["inputs"]:
                    if inp["name"] not in result:
                        failures.append((case["id"], f"missing input {inp['name']}", list(result.keys())))
                for out in case["expected"]["outputs"]:
                    if out["name"] not in result:
                        failures.append((case["id"], f"missing output {out['name']}", list(result.keys())))
            except Exception as e:
                failures.append((case["id"], "no error", str(e)))
    return "signature_parse", len(fixtures["cases"]), failures

def validate_example_ops():
    from dspy.primitives.example import Example
    with open("spec/fixtures/example_ops.json") as f:
        fixtures = json.load(f)
    failures = []
    for case in fixtures["cases"]:
        cid = case["id"]
        try:
            if case["op"] == "create":
                e = Example(**case["kwargs"])
                if sorted(e.keys()) != sorted(case["expected_keys"]):
                    failures.append((cid, case["expected_keys"], e.keys()))
                if len(e) != case["expected_len"]:
                    failures.append((cid, case["expected_len"], len(e)))
            elif case["op"] == "with_inputs":
                e = Example(**case["kwargs"]).with_inputs(*case["input_keys"])
                if sorted(e.inputs().keys()) != sorted(case["expected_inputs_keys"]):
                    failures.append((cid, case["expected_inputs_keys"], e.inputs().keys()))
                if sorted(e.labels().keys()) != sorted(case["expected_labels_keys"]):
                    failures.append((cid, case["expected_labels_keys"], e.labels().keys()))
            elif case["op"] == "contains":
                e = Example(**case["kwargs"])
                result = case["check_key"] in e
                if result != case["expected"]:
                    failures.append((cid, case["expected"], result))
            elif case["op"] == "len":
                e = Example(**case["kwargs"])
                if len(e) != case["expected"]:
                    failures.append((cid, case["expected"], len(e)))
            elif case["op"] == "to_dict":
                e = Example(**case["kwargs"])
                if e.toDict() != case["expected"]:
                    failures.append((cid, case["expected"], e.toDict()))
            elif case["op"] == "equals":
                a = Example(**case["a"])
                b = Example(**case["b"])
                if (a == b) != case["expected"]:
                    failures.append((cid, case["expected"], a == b))
            elif case["op"] == "inputs_error":
                e = Example(**case["kwargs"])
                try:
                    e.inputs()
                    failures.append((cid, "expected error", "no error"))
                except ValueError:
                    pass
        except Exception as exc:
            if "expected_error" not in case:
                failures.append((cid, "no error", str(exc)))
    return "example_ops", len(fixtures["cases"]), failures

def validate_prediction_ops():
    from dspy.primitives.prediction import Prediction
    with open("spec/fixtures/prediction_ops.json") as f:
        fixtures = json.load(f)
    failures = []
    for case in fixtures["cases"]:
        cid = case["id"]
        try:
            if case["op"] == "from_completions":
                p = Prediction.from_completions(case["completions"])
                for k, v in case["expected_store"].items():
                    if p._store.get(k) != v:
                        failures.append((cid, f"store[{k}]={v}", p._store.get(k)))
                if len(p.completions) != case["expected_completions_len"]:
                    failures.append((cid, case["expected_completions_len"], len(p.completions)))
            elif case["op"] == "float":
                p = Prediction(**case["store"])
                if "expected_error" in case:
                    try:
                        float(p)
                        failures.append((cid, "expected error", "no error"))
                    except ValueError:
                        pass
                else:
                    if float(p) != case["expected"]:
                        failures.append((cid, case["expected"], float(p)))
            elif case["op"] == "check_input_keys_absent":
                p = Prediction(**case["store"])
                if hasattr(p, "_input_keys"):
                    failures.append((cid, "absent", "present"))
        except Exception as exc:
            if "expected_error" not in case:
                failures.append((cid, "no error", str(exc)))
    return "prediction_ops", len(fixtures["cases"]), failures

if __name__ == "__main__":
    all_pass = True
    for validator in [validate_infer_prefix, validate_signature_parse, validate_example_ops, validate_prediction_ops]:
        name, total, failures = validator()
        if failures:
            all_pass = False
            print(f"FAIL: {name} — {len(failures)}/{total} failures:")
            for f in failures:
                print(f"  {f}")
        else:
            print(f"PASS: {name} — {total}/{total}")
    sys.exit(0 if all_pass else 1)
