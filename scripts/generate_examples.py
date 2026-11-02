"""Convert all example scripts into markdown files for inclusion in the docs."""

import logging
import os
import re
import subprocess
import sys
from itertools import chain
from pathlib import Path

import mkdocs_gen_files

_log = logging.getLogger(Path(__file__).name)
_log.setLevel(logging.INFO)

if not _log.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(name)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    _log.addHandler(handler)

REPOSITORY_ROOT = Path(__file__).parent.parent.parent.absolute()
EXAMPLES_DIR = REPOSITORY_ROOT / "examples"
DOCS_DIR = REPOSITORY_ROOT / "docs"
WORKFLOWS_TARGET_DIR = DOCS_DIR / "examples"
ANALYSIS_TARGET_DIR = DOCS_DIR / "post_study_analysis"
PREDICTIONS_TARGET_DIR = DOCS_DIR / "predictions"
DIAGNOSTICS_TARGET_DIR = DOCS_DIR / "diagnostics"
EXCLUDED_FILES = {"__init__.py"}
FORCE = False

_log.debug(f"Converting all scripts in {EXAMPLES_DIR} to markdown files")

WORKFLOWS_TARGET_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_TARGET_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_TARGET_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_TARGET_DIR.mkdir(parents=True, exist_ok=True)

analysis_pages: list[tuple[str, str]] = []
prediction_pages: list[tuple[str, str]] = []

for example_script in chain.from_iterable([EXAMPLES_DIR.glob("*.py"), EXAMPLES_DIR.glob("*.ipynb")]):
    if example_script.name in EXCLUDED_FILES:
        continue

    _log.info(f"Converting {example_script.name}")

    # Route files to the appropriate documentation section
    if example_script.suffix == ".py":
        target_dir = WORKFLOWS_TARGET_DIR
    elif example_script.stem.startswith("analysis_"):
        target_dir = ANALYSIS_TARGET_DIR
    elif example_script.stem == "predict_new_data":
        target_dir = PREDICTIONS_TARGET_DIR
    else:
        target_dir = DIAGNOSTICS_TARGET_DIR

    target_file = target_dir / f"{example_script.stem}.md"

    if not target_file.exists() or FORCE:
        env = os.environ | {"ALWAYS_OVERWRITE_STUDY": "yes"}

        if example_script.suffix == ".py":
            # remove a module docstring
            temp_script = target_dir / f"{example_script.stem}.py"
            with open(example_script) as inp, open(temp_script, "w", encoding="UTF-8") as out:
                module_code = inp.read()
                module_code_no_docstring = re.sub(r'^("""|\'\'\')[^"\']*\1\n*', "", module_code, flags=re.DOTALL)
                out.write(module_code_no_docstring)

            # Convert python script to notebook first
            temp_notebook = target_dir / f"{example_script.stem}.ipynb"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupytext",
                    "--to",
                    "notebook",
                    "--output",
                    str(temp_notebook),
                    str(temp_script),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                _log.error(f"✗ Failed to convert {example_script} to notebook format.\n\tstderr:\n{proc.stderr}")
                proc.check_returncode()

            example_script = temp_notebook  # noqa: PLW2901

        # execute jupyter notebook and export to markdown
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                # TODO: add "--execute", in case we want to try including results
                "-y",
                "--to",
                "markdown",
                "--embed-images",
                "--output-dir",
                str(target_file.parent),
                "--output",
                str(target_file.stem),
                str(example_script),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            _log.error(f"✗ Failed to export {example_script} to markdown format.\n\tstderr:\n{proc.stderr}")
            proc.check_returncode()

        _log.debug(f"Converted {example_script.name} to {target_file.name}")

    else:
        _log.debug(f"Not re-building existing file {target_file}")

    # Hook the file into mkdocs
    output_file = target_file.relative_to(DOCS_DIR)
    with open(target_file, encoding="UTF-8") as inp, mkdocs_gen_files.open(output_file, "w") as out:
        out.write(inp.read())

    # Track pages for generated navigation
    if target_dir == ANALYSIS_TARGET_DIR:
        analysis_pages.append((target_file.stem, f"{target_file.stem}.md"))
    elif target_dir == PREDICTIONS_TARGET_DIR:
        prediction_pages.append((target_file.stem, f"{target_file.stem}.md"))


# Generate SUMMARY.md for new sections so literate-nav can discover them
def _title(stem: str) -> str:
    """Derive a human-readable title from a file stem."""
    if stem.startswith("analysis_"):
        # "analysis_regression" -> "Regression Analysis"
        subject = stem.removeprefix("analysis_").replace("_", " ").title()
        return f"{subject} Analysis"
    return stem.replace("_", " ").title()


if analysis_pages:
    analysis_pages.sort()
    lines = [f"- [{_title(stem)}]({md})\n" for stem, md in analysis_pages]
    with mkdocs_gen_files.open("post_study_analysis/SUMMARY.md", "w") as f:
        f.writelines(lines)

if prediction_pages:
    prediction_pages.sort()
    lines = [f"- [{_title(stem)}]({md})\n" for stem, md in prediction_pages]
    with mkdocs_gen_files.open("predictions/SUMMARY.md", "w") as f:
        f.writelines(lines)
