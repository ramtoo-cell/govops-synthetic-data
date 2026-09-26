
"""Synthetic data validator.

Validates synthetic datasets against privacy and utility constraints
before they are used for training or testing:
- overlap check: no synthetic row may duplicate a real row on the
  quasi-identifier columns (strict privacy);
- range check: every value inside declared min/max bounds;
- drift check: synthetic mean and std within tolerance of the real
  distribution (utility).

Design:
- Rows are dicts; checks operate on named numeric columns.
- overlap is a fraction 0-1; the default max is 0.0: any exact
  quasi-identifier match is a privacy failure.
- Drift uses relative tolerance on mean and absolute tolerance on
  std; columns with zero real variance require exact match.
- validate() runs every check and returns a per-check report plus an
  overall pass/fail.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CheckResult:
    """Outcome of one validation check."""
    check: str
    passed: bool
    detail: str
    value: float = 0.0


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


class SyntheticDataValidator:
    """Privacy and utility checks for synthetic datasets."""

    def __init__(self, real_rows, quasi_identifiers=(), ranges=None,
                 mean_tolerance=0.1, std_tolerance=0.2):
        self.real_rows = list(real_rows)
        self.quasi_identifiers = tuple(quasi_identifiers)
        self.ranges = dict(ranges or {})
        if mean_tolerance < 0 or std_tolerance < 0:
            raise ValueError("tolerances cannot be negative")
        self.mean_tolerance = mean_tolerance
        self.std_tolerance = std_tolerance

    def check_overlap(self, synthetic_rows, max_overlap=0.0):
        """Fraction of synthetic rows matching a real row on QIDs."""
        if not self.quasi_identifiers:
            return CheckResult("overlap", True, "no quasi-identifiers", 0.0)
        real_keys = {tuple(r.get(q) for q in self.quasi_identifiers)
                     for r in self.real_rows}
        rows = list(synthetic_rows)
        if not rows:
            return CheckResult("overlap", True, "no synthetic rows", 0.0)
        hits = sum(1 for r in rows
                   if tuple(r.get(q) for q in self.quasi_identifiers) in real_keys)
        overlap = hits / len(rows)
        return CheckResult("overlap", overlap <= max_overlap,
                           "%d of %d synthetic rows match real rows"
                           % (hits, len(rows)), round(overlap, 4))

    def check_ranges(self, synthetic_rows, columns):
        """Every value in columns must fall inside declared ranges."""
        violations = []
        for row in synthetic_rows:
            for column in columns:
                if column not in self.ranges:
                    continue
                low, high = self.ranges[column]
                value = row.get(column)
                if value is None or not (low <= value <= high):
                    violations.append((column, value))
        passed = not violations
        return CheckResult("ranges", passed,
                           "%d range violations" % len(violations),
                           float(len(violations)))

    def check_drift(self, synthetic_rows, columns):
        """Synthetic mean/std must track the real distribution."""
        rows = list(synthetic_rows)
        worst = 0.0
        notes = []
        for column in columns:
            real = [r[column] for r in self.real_rows if column in r]
            synth = [r[column] for r in rows if column in r]
            if not real or not synth:
                continue
            r_mean, s_mean = _mean(real), _mean(synth)
            r_std, s_std = _std(real), _std(synth)
            mean_gap = abs(s_mean - r_mean) / abs(r_mean) if r_mean else abs(s_mean)
            std_gap = abs(s_std - r_std)
            worst = max(worst, mean_gap, std_gap)
            if mean_gap > self.mean_tolerance or std_gap > self.std_tolerance:
                notes.append(column)
        passed = not notes
        return CheckResult("drift", passed,
                           "drifted columns: %s" % ", ".join(notes)
                           if notes else "within tolerance", round(worst, 4))

    def validate(self, synthetic_rows, columns, max_overlap=0.0):
        """Run all checks; return {"passed", "checks"}."""
        rows = list(synthetic_rows)
        checks = [self.check_overlap(rows, max_overlap),
                  self.check_ranges(rows, columns),
                  self.check_drift(rows, columns)]
        return {"passed": all(c.passed for c in checks),
                "checks": {c.check: {"passed": c.passed,
                                     "detail": c.detail,
                                     "value": c.value} for c in checks}}
