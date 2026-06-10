# Welcome to torch_brain's contributing guide

Thank you for investing your time in contributing to our project! Most
contributions to `torch_brain` are made through [Pull Requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
on GitHub. We recommend first discussing the change you'd like to make by
opening a [GitHub Issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue),
especially for new features. Starting with an issue helps us align on design
choices, avoid duplicate work, and ensure your contribution fits the project's
roadmap.

## Environment Setup and Development Workflow

### Environment Setup

#### 1. Fork and Clone

First, you will need your own copy of the code to work on.

1. [Fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) the `torch_brain` repository on GitHub.
2. Clone your fork to your local machine:

```bash
git clone https://github.com/<your-username>/torch_brain
cd torch_brain
```

#### 2. Set up the Environment

We strongly recommend using a virtual environment to manage dependencies and
avoid conflicts with your system Python.

> [!NOTE]
> The instructions below use standard Python tools (`venv` and `pip`), but feel
> free to use your preferred package manager (such as **Conda**, **uv**, or
> **Poetry**) if you are comfortable with them.

Create and activate the environment:

**On macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**On Windows:**

```bash
python -m venv .venv
.venv/Scripts/activate
```

#### 3. Install for Development

Install the package in "editable" mode (`-e`). This means changes you make to
the source code are immediately reflected in your environment without needing
to reinstall the package.

```bash
pip install -e ".[dev]"
```

### Making Changes

#### 1. Create a Branch

Never work directly on the `main` branch. Always create a "feature branch" for
your specific task. This keeps your workspace clean and makes merging easier.

```bash
# Add upstream remote if you haven't already
git remote add upstream https://github.com/neuro-galaxy/torch_brain.git
# Sync with upstream first to get the latest updates
git checkout main
git pull upstream main
# Create your feature branch
git checkout -b my-username/my-new-feature
```

#### 2. Make your changes

**Where should I edit code?**

- **Core Code:** The core logic of the library is located in the `torch_brain/`
directory. This is where you will add features or fix bugs.

- **Models**: New models are added to `torch_brain/models`. If you're
adding a model, we would also encourage adding a _minimal_ training usage
in the `examples/` directory.

- **Brainsets**: New brainset pipelines go in
`torch_brain/pipeline/brainsets-pipelines`.

- **Tests:** Unit tests are located in `tests/`.

- **Documentation:** Docs are present in two places:
    - API reference docs are present in docstrings of functions/classes/modules.
    - User guides are present in `docs/source`.

#### 3. Format & Lint

All code must be formatted and linted using [Ruff](https://github.com/astral-sh/ruff).
Run ruff in your project root:

```bash
ruff format .
ruff check --fix .
```

- We recommend enabling "Format on Save" in your code editor.

- You can also set up a hook so the formatter and linter run automatically
before every commit:

```bash
pre-commit install
```

#### 4. Run Tests Locally

Before submitting, you must ensure your changes don't break existing
functionality.

- **Run the full suite:**

```bash
pytest
```

- **Run specific tests:**

```bash
pytest tests/data/test_arraydict.py
```

#### 5. Submitting Code

- When you are ready, **push** your branch and
[open a **Pull Request**](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) (PR).

- Our **Continuous Integration (CI)** Pipeline will automatically trigger. This
is a series of automated checks that run on a remote server to verify your
code.

- Write a **PR description** that documents clearly what has changed and links
to any relevant issues that your changes address.

- **Notify reviewers** directly on GitHub in the "Reviewers" panel on the
top-right of the PR page. If you are unsure who to ask for a review, you can
ask for a review from either of @vinamarora8, @AlexandreAndre, or @milosobral,
and they will tag the person appropriate to review your PR.


## Code style guidelines

We want you to feel welcome to experiment, ask questions, and enjoy building
with us. The guidelines below are here to help us stay consistent as a team,
not to slow you down. Don't hesitate to open a PR even if you're unsure about
every detail; that's what code review is for!

To ensure `torch_brain` remains maintainable and accessible, we adhere to
the following standards.

### Code Quality

- Code should be organized, easy to understand, and easy to maintain.
Prioritize **clarity over cleverness.**

- Use comments to explain why a decision was made, not what the code is doing.
  - _Rule of thumb:_ If a comment is absolutely necessary to explain the
  mechanics of your code, the code itself is likely too complex and should be
  refactored.

### Type Hints

- All new code must use Python type hints for function, method, and class
signatures.

- Please ensure your types are as specific as possible (e.g., use `list[str]`
instead of just `list` or `Any`).

### Documentation Standards

All public-facing API (functions, classes, methods, constants) **must** have
a docstring.

A good docstring looks like the following:

```python
def my_new_function(arg1: Tensor, arg2: str = "abc") -> Tensor:
    """A one-liner explanation of the function.

    A more detailed explanation. You can talk about some important
    use cases, or implementation details that would matter during this
    function's use.

    .. note::

       Write any edge-cases as notes

    Args:
        arg1: short description for argument 1 :math:`(N, D)`
            # use :math: to describe shape expectations for arrays/tensors
        arg2: short description for argument 2 (default "abc").
            # always write default value in the docstring

    Returns:
        Concise description of what this function returns

    Shape legend:  # always good-to-have when working with tensors
        - :math:`N`: Embedding sequence length
        - :math:`D`: Hidden dimension of the embeddings
    """
```

- Docstrings should concisely describe the component's purpose, arguments,
return values, and important shape conventions (for tensors/arrays).

- Every argument must be listed in the docstring with its type and a
description of its role.

- If a function is non-trivial (i.e., it involves complex logic, specific
tensor shapes, or edge cases), provide a functional usage example.
  - Examples should be self-contained and runnable.

### Naming Conventions

We follow standard Python (PEP 8) naming conventions. Please ensure your
identifiers follow this pattern:

- **Functions & Variables:** `snake_case`

- **Classes:** `PascalCase`

- **Constants:** `UPPER_SNAKE_CASE`

- **Protected/Private Members:** Prefix with a single underscore
`_variable_name` to indicate internal use.


## Design guidelines

### PyTorch-friendly workflows

- Features and APIs that affect general user workflows should follow PyTorch's
design philosophy and stay compatible with common usage patterns.

- Naming should intuitively make sense to anyone familiar with the ecosystem.
If a function is analogous to a standard PyTorch or NumPy operation, use the
established name.

- Deviations are acceptable only when they provide clear benefits for our
specialized use cases. Any deviation is expected to remain fully optional, with
a clear path for users who prefer standard PyTorch workflows.

### Abstractions and Class Hierarchies

- Use abstractions or class hierarchies only when they genuinely help by making
the code clearer, reducing duplication, or improving reuse.

- Abstractions must not lock users into a rigid workflow. If an abstraction
makes it harder to access underlying data or modify behavior, it is likely too
restrictive.

- Avoid over-engineering. An abstraction is invalid if it significantly
steepens the learning curve for new users. The goal is to reduce cognitive
load, not increase it.

- Prefer simple, direct designs over deep inheritance trees. Composition is
often preferred over inheritance.

### Testing Standards

- All new features must include unit tests. Major architectural changes must
maintain or improve existing coverage.

- Every bug fix must include a regression test that fails without the fix and
passes with it.

- Tests should be isolated and fast. Use descriptive names (e.g.,
`test_forward_pass_invalid_dim`) and rely on `torch.testing.assert_close` for
tensor comparisons to handle floating-point precision.

### Duck typing

- APIs are encouraged to accept any object that provides the needed interface
(e.g., PyTorch tensors, NumPy arrays, array-like objects) rather than enforcing
strict types.

- When duck typing is used, document the expected behavior clearly (e.g., must
support indexing or `.shape`).

- Input validation should focus on the capabilities of objects, not their exact
inheritance lineage.

### API deprecations

- As a growing ecosystem, deprecations are expected but must be handled
gracefully.

- When modifying or removing a long-standing API, include a clear
`DeprecationWarning`. This warning should explain what is changing and provide
a direct migration path for the user to the new API.

### `device` and `dtype` behavior

- Respect the `device` and `dtype` of input tensors.

- Do not hard-code CUDA usage or specific `dtypes` in library code.

- As a rule, all code should run seamlessly on both CPU and GPU (and other
accelerators) whenever the underlying operations support it.

## Python environment

- We support installation through standard Python tooling, and `torch_brain`
must remain fully compatible with `pip install torch_brain`. Any change that
breaks or complicates pip-based installation is not allowed.

- Contributors may use other tools (such as `uv`, `conda`, `poetry`, etc) for
their local workflow, but the codebase and documentation must not depend on
them.
