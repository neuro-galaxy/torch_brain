# Releasing to PyPI

1. Ensure `main` is ready to release.
2. Make sure `CHANGELOG.md` is up to date (review commit history to verify, open a Release PR if changes are needed).
3. Verify the version in `CHANGELOG.md` matches the version you're about to tag.
4. Create and push the release tag ([see here](#valid-version-numbers-and-their-meaning) for version syntax).

```bash
$ git checkout main
$ git pull origin main
$ git tag vX.Y.Z
$ git push origin vX.Y.Z
```

5. Monitor the GitHub Actions workflow to ensure the build and publish succeed.
6. Add a GitHub Release with a small description of changes from the last release.

**Note:** Tags must start with "v" for the publishing GitHub Action to begin.

## Valid version numbers and their meaning

- For version number, we follow [SemVer](https://semver.org/) (major.minor.patch).
- For pre-release tags, we follow the [PEP440](https://peps.python.org/pep-0440/) syntax:
```
vX.Y.ZaN   # Alpha release
vX.Y.ZbN   # Beta release
vX.Y.ZrcN  # Release candidate
vX.Y.Z     # Final release
```
- For post-release tags, we also follow the [PEP440](https://peps.python.org/pep-0440/) syntax:
  - `vX.Y.Z.postN`, `vX.Y.ZaN.postM`, ...
  - Post releases are only meant for metadata/distribution related corrections, and not code edits
