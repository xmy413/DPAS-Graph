
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional
import re
from collections import defaultdict

import anndata as ad


_nt_re = re.compile(r"^[ACGT]+$", re.IGNORECASE)
_suffix_digits_re = re.compile(r"^(.*)_(\d+)$")
_safe_base_re = re.compile(r"^[A-Za-z0-9]+$")


def strip_total_seq_prefix(name: str) -> str:
    s = str(name).strip()
    for pref in ("TotalSeq-A_", "TotalSeq_A_", "TotalSeq-C_", "TotalSeq_C_"):
        if s.startswith(pref):
            s = s[len(pref):]
    return s


def strip_oligo_suffix(name: str, min_nt_len: int = 6) -> str:
    s = str(name).strip()
    for sep in ("-", ".", "_"):
        if sep in s:
            left, right = s.rsplit(sep, 1)
            r = right.strip()
            if len(r) >= min_nt_len and _nt_re.match(r):
                return left.strip()
    return s


def clean_protein_name_list(var_names: Sequence[str], min_nt_len: int = 6) -> List[str]:
    name_groups = defaultdict(list)
    parsed = []
    all_cleaned = set()

    for vn in var_names:
        s = strip_total_seq_prefix(vn)
        s = strip_oligo_suffix(s, min_nt_len=min_nt_len)
        s = s.replace("-", "_").replace(".", "_")
        all_cleaned.add(s)

        m = _suffix_digits_re.match(s)
        if m:
            base, idx_str = m.group(1), m.group(2)
            parsed.append((base, idx_str, s))
            name_groups[base].append(s)
        else:
            parsed.append((s, None, s))
            name_groups[s].append(s)

    out: List[str] = []
    for base, idx_str, original in parsed:
        if idx_str is not None:
            can_fold = (
                len(name_groups[base]) == 1
                and _safe_base_re.match(base) is not None
                and (base not in all_cleaned)
            )
            out.append(base if can_fold else original)
        else:
            out.append(original)
    return out


def is_isotype(name: str) -> bool:
    sl = str(name).strip().lower()
    return (
        sl.startswith("mouse_") or sl.startswith("rat_") or
        sl.startswith("mouse.") or sl.startswith("rat.") or
        ("isotype" in sl) or ("control" in sl)
    )


def dedup_keep_first(names: Sequence[str]) -> Tuple[List[str], List[bool]]:
    seen = set()
    keep = []
    out = []
    for n in names:
        if n in seen:
            keep.append(False)
        else:
            seen.add(n)
            keep.append(True)
            out.append(n)
    return out, keep

def load_specs(pairs_json: str) -> List[Dict]:
    obj = json.loads(Path(pairs_json).read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "specs" in obj:
        obj = obj["specs"]
    if not isinstance(obj, list):
        raise ValueError("pairs_json must be a list or {specs:[...]}")

    specs = []
    for s in obj:
        name = s.get("name") or s.get("dataset") or s.get("id")
        if name is None:
            raise ValueError("each spec must have a 'name'")
        adt_path = s.get("adt") or s.get("adt_h5ad") or s.get("adt_path")
        if adt_path is None:
            raise ValueError(f"spec {name} missing adt path")
        specs.append({"name": str(name), "adt": str(adt_path)})
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_json", required=True)
    ap.add_argument("--proteins_out", required=True)
    ap.add_argument("--audit_out", default="")
    ap.add_argument("--remove_isotype", action="store_true")
    ap.add_argument("--min_nt_len", type=int, default=6)
    args = ap.parse_args()

    specs = load_specs(args.pairs_json)

    cleaned_names_per_ds: List[List[str]] = []
    ds_order_ref: Optional[List[str]] = None
    audit_rows: List[str] = ["name,n_in,n_after_isotype,n_after_dedup,n_renamed,n_removed_isotype,n_removed_dup"]

    for sp in specs:
        name = sp["name"]
        adt = ad.read_h5ad(sp["adt"], backed="r")  

        raw = list(map(str, adt.var_names))
        cleaned = clean_protein_name_list(raw, min_nt_len=args.min_nt_len)

        n_renamed = sum(int(a != b) for a, b in zip(raw, cleaned))

 
        keep1 = [True] * len(cleaned)
        if args.remove_isotype:
            keep1 = [not is_isotype(x) for x in cleaned]
        cleaned2 = [x for x, k in zip(cleaned, keep1) if k]
        n_after_isotype = len(cleaned2)
        n_removed_isotype = len(cleaned) - n_after_isotype if args.remove_isotype else 0

        # dedup keep first
        cleaned3, keep2 = dedup_keep_first(cleaned2)
        n_after_dedup = len(cleaned3)
        n_removed_dup = n_after_isotype - n_after_dedup

        cleaned_names_per_ds.append(cleaned3)

        if ds_order_ref is None:
            ds_order_ref = cleaned3

        audit_rows.append(f"{name},{len(raw)},{n_after_isotype},{n_after_dedup},{n_renamed},{n_removed_isotype},{n_removed_dup}")

    # intersection after cleaning
    common = set(ds_order_ref)
    for lst in cleaned_names_per_ds[1:]:
        common &= set(lst)
    if len(common) == 0:
        raise RuntimeError("No common proteins across datasets after cleaning; check cleaning rules / panels.")

    # preserve order from first dataset
    proteins = [p for p in ds_order_ref if p in common]

    Path(args.proteins_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.proteins_out).write_text("\n".join(proteins), encoding="utf-8")

    if args.audit_out:
        Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_out).write_text("\n".join(audit_rows), encoding="utf-8")

    print(f"[Stage1] wrote proteins_clean_common: P={len(proteins)} -> {args.proteins_out}")
    if args.audit_out:
        print(f"[Stage1] wrote audit csv -> {args.audit_out}")


if __name__ == "__main__":
    main()
