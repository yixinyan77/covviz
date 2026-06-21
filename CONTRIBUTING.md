# Contributing

Thanks for helping improve `covviz`.

## Local Setup

```bash
git clone https://github.com/yixinyan77/covviz.git
cd covviz
pip install -e ".[dev]"
```

## Run Tests

```bash
python -m pytest
```

## Development Notes

- Keep the public API simple: prefer adding options to `covviz.plot` before
  creating new top-level functions.
- Plot functions should accept an optional Matplotlib `ax` and return the axis.
- Matrix validation should remain strict by default: inputs must be square,
  finite, and symmetric.
- Avoid adding heavy dependencies to the core package. Put domain-specific
  integrations behind optional extras.
