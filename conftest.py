"""Pytest bootstrap.

1. Put the flat `code/` module directory on sys.path so the experiment modules
   (canonicalize, certify, stats, blackbox_detectors, npl_adaptive_experiment_v2,
   ...) import the same way the experiments and the tests in tests/ do. Tests that
   need the GPU-only stack (torch / sentence-transformers) import it lazily and are
   skipped by their own guards when it is absent; nothing here masks a real failure.

2. The `code/` directory is a package (so `python -m code.reproduce` works), but its
   name collides with the standard library `code` module that pytest's pdb plugin
   imports (`code.InteractiveConsole`). During tests we therefore pin the REAL stdlib
   `code` module into sys.modules. Tests import the experiment modules flat (via the
   path insertion above), never as `code.<x>`, so this pin is invisible to them and
   only affects the separate `python -m code.reproduce` process not at all (conftest
   is not loaded there)."""
import importlib.util
import os
import sys
import sysconfig

_CODE = os.path.join(os.path.dirname(__file__), "code")
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

# Pin stdlib `code` so pytest/pdb internals resolve InteractiveConsole correctly.
if not hasattr(sys.modules.get("code"), "InteractiveConsole"):
    _stdlib_code = os.path.join(sysconfig.get_path("stdlib"), "code.py")
    if os.path.exists(_stdlib_code):
        _spec = importlib.util.spec_from_file_location("code", _stdlib_code)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        sys.modules["code"] = _mod
