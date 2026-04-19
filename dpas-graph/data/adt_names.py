from dataclasses import dataclass
from typing import List, Sequence
import re
from collections import defaultdict

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

@dataclass
class ProteinCleanReport:
    n_in: int
    n_after_isotype: int
    n_after_dedup: int
    n_renamed: int
    n_removed_isotype: int
    n_removed_dup: int

def clean_adt_varnames_inplace(
    adt,
    *,
    remove_isotype: bool = True,
    min_nt_len: int = 6,
    keep_first_duplicate: bool = True,
) -> ProteinCleanReport:
    if "protein_raw" not in adt.var.columns:
        adt.var["protein_raw"] = list(map(str, adt.var_names))

    raw_names = list(map(str, adt.var_names))
    cleaned = clean_protein_name_list(raw_names, min_nt_len=min_nt_len)

    adt.var_names = cleaned
    adt.var["protein_clean"] = list(map(str, adt.var_names))
    n_renamed = sum(int(a != b) for a, b in zip(raw_names, cleaned))

    n_in = len(raw_names)

    if remove_isotype:
        keep = [not is_isotype(n) for n in adt.var_names]
        adt._inplace_subset_var(keep)

    n_after_isotype = adt.n_vars
    n_removed_isotype = n_in - n_after_isotype if remove_isotype else 0

    if keep_first_duplicate:
        seen = set()
        keep2 = []
        for n in map(str, adt.var_names):
            if n in seen:
                keep2.append(False)
            else:
                seen.add(n)
                keep2.append(True)
        if sum(keep2) < len(keep2):
            adt._inplace_subset_var(keep2)

    n_after_dedup = adt.n_vars
    n_removed_dup = n_after_isotype - n_after_dedup

    return ProteinCleanReport(
        n_in=n_in,
        n_after_isotype=n_after_isotype,
        n_after_dedup=n_after_dedup,
        n_renamed=n_renamed,
        n_removed_isotype=n_removed_isotype,
        n_removed_dup=n_removed_dup,
    )