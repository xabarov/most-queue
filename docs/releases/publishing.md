# Publishing Most-Queue

Publication is a manual maintainer action. A GitHub Release or Git tag does not
mean the version exists on PyPI; check the [PyPI release history](https://pypi.org/project/most-queue/#history) separately.

## Trusted Publisher setup

An owner of the existing `most-queue` PyPI project must add a GitHub Trusted
Publisher in its Publishing settings using these exact fields:

| Field | Value |
| --- | --- |
| GitHub owner | `xabarov` |
| Repository name | `most-queue` |
| Workflow filename | `publish.yml` |
| Environment name | `pypi` |

Create the `pypi` environment in the GitHub repository settings. Configure required
reviewers and allowed deployment refs according to the maintainers' policy.
Allow the ref selected when dispatching the workflow, normally `main`.
Authentication uses OpenID Connect; no PyPI password or API-token secret is needed.
Only the publishing job has `id-token: write`, following the official
[PyPI instructions](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Prepare and publish

1. Update `pyproject.toml` and `CITATION.cff` to the same new version, prepare
   release notes, run required project checks, and review the changes.
2. Commit and push the reviewed changes, including this workflow on the default
   branch. Create and push a new tag such as `v2.11` at the release commit.
   **Never move or replace an existing tag.** Retain `v2.10` at its original commit.
3. Open **Actions → Publish to PyPI → Run workflow**, select the reviewed workflow
   ref (normally `main`), and enter `2.11` in `version`, without `v`.
   Only numeric versions with two or three components are accepted.
4. The build job checks out exactly `refs/tags/v2.11`, verifies the commit and both
   version fields, builds wheel and source distributions, and runs `twine check`.
   It installs the wheel in a fresh virtual environment outside the checkout and
   checks imports and calculations for `MMnrCalc` and `TransientOccupancyQueueCalc`
   using isolated Python mode.
5. Checked distributions are saved as an Actions artifact. Review the run and
   approve the `pypi` environment deployment if configured. The publishing job
   downloads that exact artifact by ID and uploads it without rebuilding.
6. Confirm the version and files on PyPI, then check installation in a clean
   environment. Create a GitHub Release and its notes separately if desired.

Pushing a tag or creating a GitHub Release does not trigger publication.
Dispatching this workflow authorizes publication once its checks and configured
environment requirements pass. Do not rerun publication for an uploaded version:
PyPI does not permit replacing distribution files. Corrections require a new
version and a new tag.

## Local packaging checks

From a clean checkout of the intended tag, in a virtual environment:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

These commands do not publish. The manual workflow also checks the installed wheel.
