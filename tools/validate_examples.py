#!/usr/bin/env python3
"""
validate_examples.py
Validate JSON outputs inside a second-train.txt style file against a JSON Schema.
Usage:
  python tools/validate_examples.py --schema schema.json --input second-train.txt
"""

import argparse
import json
import re
from jsonschema import validate, ValidationError, Draft7Validator

BLOCK_RE = re.compile(r"###\s*Instruction\s*(.*?)\s*###\s*Input\s*(.*?)\s*###\s*Output\s*(\{.*?\})(?=\n\n|$)", re.S)

def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_examples(path):
    text = open(path, "r", encoding="utf-8").read()
    matches = BLOCK_RE.findall(text)
    examples = []
    for inst, inp, out in matches:
        out = out.strip()
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError as e:
            examples.append({"error": "json_decode_error", "detail": str(e), "raw": out})
            continue
        examples.append({"instruction": inst.strip(), "input": inp.strip(), "output": parsed})
    return examples

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schema", required=True)
    p.add_argument("--input", required=True)
    args = p.parse_args()

    schema = load_schema(args.schema)
    examples = parse_examples(args.input)

    validator = Draft7Validator(schema)
    errors = []
    for i, ex in enumerate(examples, start=1):
        if "error" in ex:
            errors.append((i, "invalid_json", ex["detail"], ex["raw"]))
            continue
        out = ex["output"]
        errs = sorted(validator.iter_errors(out), key=lambda e: e.path)
        if errs:
            errors.append((i, "schema_errors", [e.message for e in errs], out))

    if errors:
        print(f"Validation failed: {len(errors)} problematic examples found.")
        for idx, kind, detail, raw in errors:
            print(f"\nExample #{idx} - {kind}")
            if isinstance(detail, list):
                for d in detail:
                    print(f"  - {d}")
            else:
                print(f"  {detail}")
            print(f"  Output: {raw}")
        raise SystemExit(2)
    else:
        print(f"All {len(examples)} examples validated successfully.")
        raise SystemExit(0)

if __name__ == "__main__":
    main()
