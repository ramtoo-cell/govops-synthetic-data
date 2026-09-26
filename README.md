# Synthetic Data Validator

Validates synthetic datasets for privacy overlap, value ranges, and distribution drift against real data.

**Stack:** Python 3.12 · stdlib only · unittest

## Problem

Validates synthetic datasets against privacy and utility constraints
before they are used for training or testing:
- overlap check: no synthetic row may duplicate a real row on the
  quasi-identifier columns (strict privacy);
- range check: every value inside declared min/max bounds;
- drift check: synthetic mean and std within tolerance of the real
  distribution (utility).

## Design decisions

- Rows are dicts; checks operate on named numeric columns.
- overlap is a fraction 0-1; the default max is 0.0: any exact
- Drift uses relative tolerance on mean and absolute tolerance on
- validate() runs every check and returns a per-check report plus an

## Layout

- `README.md`
- `src/synthetic_data.py`
- `tests/test_synthetic_data.py`
- `LICENSE`
- `.gitignore`

## Setup

No third-party dependencies — Python 3.11+ standard library only.

```bash
cd govops-synthetic-data
python -m unittest discover -s tests -v
```

## Usage

```python
from src.synthetic_data import SyntheticDataValidator

real = [{"age": 30, "zip": "10001"}, {"age": 40, "zip": "10002"}]
v = SyntheticDataValidator(real, quasi_identifiers=("age", "zip"))
print(v.validate([{"age": 31, "zip": "10001"}], ["age"])["passed"])
```

## License

MIT — see [LICENSE](LICENSE).
