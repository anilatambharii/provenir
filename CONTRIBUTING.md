# Contributing to Provenir

Thank you for contributing! This document covers the development workflow,
code standards, and PR process.

---

## Development Setup

```bash
git clone https://github.com/anilatambharii/provenir
cd provenir
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,serve,hub,merge,semantic,judge-anthropic,judge-openai]"
```

Requires Python â‰¥ 3.11.

---

## Quality Gate

Every PR must pass three checks. Run them locally before pushing:

```bash
python -m ruff check .        # linting
python -m mypy src            # type checking (strict)
python -m pytest -q           # test suite (456 tests)
```

CI runs these automatically on push and PR. A red CI is a blocker.

---

## Code Standards

### Style

- Line length: 100 characters (enforced by ruff)
- Imports: sorted by ruff (`I` rule), stdlib â†’ third-party â†’ local
- Type annotations: required on all public functions and methods (mypy strict)

### Comments

Write **no comments** by default. Add one only when the *why* is non-obvious:
a hidden constraint, a subtle invariant, or a workaround for a specific bug.
Never describe what the code does â€” well-named identifiers already do that.

### Error handling

Validate only at system boundaries (user input, external APIs). Trust internal
code and framework guarantees. Do not add defensive checks for scenarios that
can't happen in correct usage.

### Optional dependencies

All optional packages (torch, TRL, anthropic, etc.) must be guarded with the
conditional import pattern:

```python
try:
    import some_package
    _HAS_SOME_PACKAGE = True
except ImportError:
    _HAS_SOME_PACKAGE = False
```

Provide a stub implementation that degrades gracefully when the package is
absent. Every feature must work (with stubs) in a `pip install provenir`
environment with no optional extras.

---

## Testing

- Every new feature needs tests in `tests/`
- Test files are named `test_<module>.py`
- Tests that exercise stub behaviour must toggle the `_HAS_X` flag:

```python
def setup_method(self) -> None:
    import provenir.some.module as mod
    self._orig = mod._HAS_SOMETHING
    mod._HAS_SOMETHING = False

def teardown_method(self) -> None:
    import provenir.some.module as mod
    mod._HAS_SOMETHING = self._orig
```

Use `tmp_path` (pytest fixture) for any file I/O in tests. Never write to
a fixed path â€” tests run in parallel and on shared CI machines.

---

## PR Process

1. **Fork** the repository and create a branch from `main`
2. **Implement** your change â€” keep the scope focused
3. **Run the quality gate** locally (ruff + mypy + pytest)
4. **Open a PR** â€” title should be a concise imperative sentence
5. A maintainer will review and request changes or merge

Branch naming convention:

| Type | Pattern |
|---|---|
| Feature | `feat/short-description` |
| Bug fix | `fix/short-description` |
| Docs | `docs/short-description` |
| Refactor | `refactor/short-description` |

---

## What to Work On

Good first issues are labelled [`good first issue`](https://github.com/anilatambharii/provenir/labels/good%20first%20issue) on GitHub.

Larger contributions (new backends, new eval metrics, new CLI commands) should
start with a GitHub Issue describing the approach before implementation begins.

---

## License

By submitting a PR you agree that your contribution will be licensed under the
[Apache 2.0 license](LICENSE).

