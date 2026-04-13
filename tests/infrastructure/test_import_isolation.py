"""Import isolation tests for the predict package.

These tests ensure that ``import octopus.predict`` does NOT pull in
heavy or unnecessary dependencies that are only needed for running
studies.  This guards long-term version stability of the deployment
interface (OctoPredictor / OctoTestEvaluator).

Each test runs in a **subprocess** so the import state is clean and
unaffected by whatever the test suite has already loaded.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from typing import Any, ClassVar

import pytest

# ── Helpers ─────────────────────────────────────────────────────


def _run_import_check(code: str) -> dict[str, Any]:
    """Run a Python snippet in a subprocess and return parsed JSON output."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
    return json.loads(result.stdout)  # type: ignore[no-any-return]


# ── Tests ───────────────────────────────────────────────────────


class TestPredictImportIsolation:
    """Guard tests: ``import octopus.predict`` must NOT load study-only packages."""

    # Packages that the predict layer must NEVER import at import time.
    # These are study-execution dependencies, not prediction dependencies.
    FORBIDDEN_PACKAGES: ClassVar[list[str]] = [
        "catboost",
        "xgboost",
        "optuna",
        "networkx",
    ]

    # Octopus sub-packages that must NOT be imported by octopus.predict.
    FORBIDDEN_OCTOPUS_PACKAGES: ClassVar[list[str]] = [
        "octopus.models",
        "octopus.modules",
        "octopus.manager",
        "octopus.study",
        "octopus.datasplit",
    ]

    def test_no_forbidden_third_party_packages(self) -> None:
        """Importing octopus.predict must not load catboost, xgboost, optuna, or networkx."""
        code = f"""\
            import json, sys
            import octopus.predict
            forbidden = {self.FORBIDDEN_PACKAGES!r}
            loaded = {{}}
            for pkg in forbidden:
                modules = [m for m in sys.modules if m == pkg or m.startswith(pkg + ".")]
                loaded[pkg] = modules
            print(json.dumps(loaded))
        """
        loaded = _run_import_check(code)

        violations = {pkg: mods for pkg, mods in loaded.items() if mods}
        if violations:
            detail = "\n".join(
                f"  {pkg}: {len(mods)} modules loaded (e.g. {mods[0]})" for pkg, mods in violations.items()
            )
            pytest.fail(
                f"import octopus.predict loaded forbidden packages:\n{detail}\n"
                "These packages are study-execution dependencies and must not "
                "be imported by the predict layer."
            )

    def test_no_forbidden_octopus_subpackages(self) -> None:
        """Importing octopus.predict must not load octopus.models, .modules, .manager, .study, or .datasplit."""
        code = f"""\
            import json, sys
            import octopus.predict
            forbidden = {self.FORBIDDEN_OCTOPUS_PACKAGES!r}
            loaded = {{}}
            for pkg in forbidden:
                modules = [m for m in sys.modules if m == pkg or m.startswith(pkg + ".")]
                loaded[pkg] = modules
            print(json.dumps(loaded))
        """
        loaded = _run_import_check(code)

        violations = {pkg: mods for pkg, mods in loaded.items() if mods}
        if violations:
            detail = "\n".join(
                f"  {pkg}: {len(mods)} modules ({', '.join(mods[:3])}{'...' if len(mods) > 3 else ''})"
                for pkg, mods in violations.items()
            )
            pytest.fail(
                f"import octopus.predict loaded forbidden octopus sub-packages:\n{detail}\n"
                "The predict layer should only depend on octopus.types, "
                "octopus.utils, octopus.metrics, and octopus.predict."
            )

    def test_octopus_module_count_bounded(self) -> None:
        """The number of octopus modules loaded by import octopus.predict should stay bounded.

        This is a soft guard — increasing the count is allowed when justified,
        but unintentional additions (e.g. new transitive imports) will be caught.
        """
        code = """\
            import json, sys
            import octopus.predict
            octopus_modules = sorted(m for m in sys.modules if m.startswith("octopus"))
            print(json.dumps({"count": len(octopus_modules), "modules": octopus_modules}))
        """
        result = _run_import_check(code)

        max_allowed = 16  # Exact target: see doc 16_predict_import_isolation.md
        if result["count"] > max_allowed:
            modules_list = "\n  ".join(result["modules"])
            pytest.fail(
                f"import octopus.predict loaded {result['count']} octopus modules "
                f"(max allowed: {max_allowed}):\n  {modules_list}\n"
                "If this increase is intentional, update max_allowed in this test."
            )
