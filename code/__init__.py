"""HybridGuard / Certified-Canonicalization experiment package.

The modules in this directory are written as a FLAT module set (they import each
other as `import certify`, `import stats`, ...). When the package is executed as
`python -m code.reproduce`, reproduce.py prepends this directory to sys.path so the
flat imports resolve identically to how the experiments and tests run.
"""
