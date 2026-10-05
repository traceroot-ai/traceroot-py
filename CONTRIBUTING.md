# Contributing

Thanks for contributing to `traceroot-py`.

## Requirements

- Python 3.11, 3.12 or 3.13
- `uv`
- `git`

## Setup

```bash
git clone https://github.com/traceroot-ai/traceroot-py.git
cd traceroot-py
```

## Before you start

- Check for an existing issue before starting larger work, or open one first so the change has clear scope.
- Create branches from `main`. If you don't have push access, fork first.
- Keep each pull request focused on one problem.

## Workflow

1. Create a branch from `main` using a short descriptive name (e.g. `feat/add-openai-instrumentation`, `fix/span-flush-timeout`).
2. Make the smallest change that fully solves the issue.
3. Run linting and tests locally before pushing.
4. Open a pull request and link the related issue when applicable.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add google genai instrumentation
fix: handle missing git remote gracefully
docs: expand contributing guide
test: cover disabled client initialization
chore: bump development dependencies
```

Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `build`.

## Pull requests

- Keep PRs scoped to one logical change.
- Explain what changed, why it changed, and how it was validated.
- Add or update tests for behavior changes.
- Update documentation for API or behavior changes.
- Make sure linting and tests pass before requesting review.
