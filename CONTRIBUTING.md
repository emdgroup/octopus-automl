# Contributing to Octopus

**All contributions to Octopus are welcome!** Bug fixes, new features, docs improvements, typo corrections - everything helps.

## Quick Start

1. Fork and clone the repository

2. Set up your development environment (Python 3.12 only):

   ```bash
   conda create -n octopus-dev python=3.12
   conda activate octopus-dev
   uv sync --extra dev
   ```

   **Note:** We use [uv](https://docs.astral.sh/uv/) for fast and reliable dependency management.

3. Install pre-commit hooks:

   ```bash
   pre-commit install
   pre-commit install --hook-type commit-msg
   ```

4. Create a feature branch:

   ```bash
   git checkout -b <type>/<issue>_<description>
   # Example: git checkout -b feat/123_add-ensemble-selection
   ```

## Development Workflow

- Make your changes
- Run tests and quality checks (see below)
- Update CHANGELOG.md
- Commit with semantic message
- Push and create PR

## Package Management

We use [uv](https://docs.astral.sh/uv/) for dependency management. All dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

### Adding a New Package

```bash
uv add <package-name>
# Example: uv add pandas
```

### Updating the Lock File

After manually editing `pyproject.toml`, update the lock file:

```bash
uv lock
```

### Syncing Your Environment

After pulling changes from the repository:

```bash
uv sync --extra dev
```

This installs all dependencies according to the lock file, ensuring everyone has the same versions.

## Testing and Quality

- Run tests:

  ```bash
  pytest
  ```

- Run specific test module:

  ```bash
  pytest tests/data/test_validator.py -v
  ```

- Run with coverage:

  ```bash
  pytest --cov=octopus --cov-report=html
  ```

- Run all quality checks:

  ```bash
  pre-commit run --all-files
  ```

- Individual tools:

  ```bash
  ruff check .      # Linting
  ruff format .     # Formatting
  typos             # Spell checking
  ```

## Branch Naming

Format: `<type>/<issue>_<slug>`

**Valid types:**

- `feat` - New features
- `fix` - Bug fixes
- `docs` - Documentation changes
- `style` - Formatting, missing semicolons, etc. (no code change)
- `refactor` - Code restructuring without changing behavior
- `test` - Adding or updating tests
- `chore` - Maintenance tasks, dependency updates
- `perf` - Performance improvements
- `ci` - CI/CD configuration changes
- `build` - Build system or external dependency changes
- `revert` - Reverting previous commits

**Examples:**

- ✓ `feat/90_add-ensemble-selection`
- ✓ `fix/124_memory-leak`
- ✓ `docs/update-readme`
- ✓ `ci/138_add-pre-commit-hooks`
- ✗ `Add-New-Feature` (wrong format)

## Commit Messages

Format: `<type>: <description>`

**Valid types:** See [Branch Naming](#branch-naming) section for complete list of types and their descriptions.

**Examples:**

- ✓ `feat: add ensemble selection method`
- ✓ `fix: resolve memory leak in data loader`
- ✓ `docs: update installation guide`
- ✓ `ci: add pre-commit validation hooks`

**Auto-close issues:**

- `fixes #123` → for bugs
- `resolves #123` → for features
- `closes #123` → for tasks/docs

**Full example:**

```bash
git commit -m "feat: add ensemble selection resolves #90"
```

## CHANGELOG.md

**Required:** Each PR must update CHANGELOG.md

1. Add entry under `## [Unreleased]`

2. Use appropriate section:

   - **Added** - New features
   - **Changed** - Changes to existing functionality
   - **Deprecated** - Soon-to-be removed features
   - **Removed** - Removed features
   - **Fixed** - Bug fixes
   - **Security** - Vulnerability fixes

3. Format:

   ```markdown
   ## [Unreleased]

   ### Added

   - New ensemble selection method for improved performance (#123)

   ### Fixed

   - Memory leak in data loader (#124)
   ```

**Tips:**

- Write from user perspective
- Reference issue/PR number
- Be concise but clear

## Code Style

### Docstrings

Follow [Google Style Guide](https://google.github.io/styleguide/pyguide.html):

```python
def example_function(arg1: str, arg2: int) -> bool:
    """Short one-line summary.

    Optional longer description if needed.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something goes wrong.
    """
    ...
```

**Notes:**

- Use type hints (don't repeat them in docstrings)
- One-line summary first
- Use double backticks for literals: ` `MyString` `

### attrs Classes

```python
from attrs import define

@define
class DataConfig:
    """Configuration for data processing."""

    n_samples: int
    """Number of samples in the dataset."""

    n_features: int
    """Number of features in the dataset."""

    @n_samples.validator
    def _validate_n_samples(self, attribute, value):
        """Validate n_samples is positive."""
        if value <= 0:
            raise ValueError("n_samples must be positive")
```

**Conventions:**

- Attribute docstrings below the declaration
- Blank line between attributes
- Name validators: `_validate_<attribute_name>`
- Name defaults: `_default_<attribute_name>`

## Package Structure

```
octopus/
├── data/       # Data handling and validation
├── models/     # Model definitions and wrappers
├── modules/    # Feature selection and optimization
├── metrics/    # Performance metrics
└── config/     # Configuration management
```

When adding new functionality:

- Follow existing package structure
- Import public APIs into high-level namespaces
- Consider small dataset optimization (<1k samples)

## Pull Request Guidelines

**Main branch stability:**
- All commits on the `main` branch should be stable
- PRs are squash-merged or rebased to maintain clean history

**Commit organization:**
- Multiple commits in a PR are allowed
- Should be consolidated into a reasonable contribution
- Each commit should be logical and buildable
- Avoid WIP commits, fixups, or "oops" commits

**Good PR structure:**
```
✓ feat/123_add-feature
  - Add core functionality
  - Add tests
  - Update documentation

✗ feat/123_add-feature
  - WIP initial try
  - fix typo
  - oops forgot file
  - actually fix it
  - revert previous
```

**Tips:**
- Squash small fixups before submitting
- Use interactive rebase to clean up history: `git rebase -i main`
- Each commit should pass tests
- Maintainers may squash-merge if needed

## Syncing Your PR

If the main branch has moved ahead:

```bash
# Fetch latest changes
git fetch upstream

# Rebase (recommended for clean history)
git rebase upstream/main

# Or merge (if rebase is too complex)
git merge upstream/main
```

**Note:** We prefer rebase for linear history, but may squash-merge your PR if needed.

## Developer Tools

| Tool                                               | Purpose                     |
| -------------------------------------------------- | --------------------------- |
| [uv](https://docs.astral.sh/uv/)                   | Dependency management       |
| [ruff](https://docs.astral.sh/ruff/)               | Linting and formatting      |
| [pydoclint](https://github.com/jsh9/pydoclint)     | Docstring checking          |
| [pyupgrade](https://github.com/asottile/pyupgrade) | Python syntax upgrading     |
| [typos](https://github.com/crate-ci/typos)         | Spell checking              |
| [pytest](https://docs.pytest.org/)                 | Testing                     |
| [pytest-cov](https://pytest-cov.readthedocs.io/)   | Test coverage               |
| [pre-commit](https://pre-commit.com/)              | Git hooks orchestration     |

All tools run automatically via pre-commit hooks and CI/CD.

## Questions?

- Open an issue: [GitHub Issues](https://github.com/emdgroup/octopus-automl/issues/)
- Contact maintainers: [CONTRIBUTORS.md](CONTRIBUTORS.md)

**Thank you for contributing to Octopus!**
