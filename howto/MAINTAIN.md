# Maintaining `torch_brain`

Notes and guidelines on maintaining the `torch_brain` project.

## Commit messages

Since we mostly perform "Squash & Merge," the title of the PR forms the main
commit message. In addition, things are set up so that the PR description forms
the default commit message description. Thus, to quality-control what the
commit messages are, we have to quality-control the PR title and descriptions.

Of these, the most important is that the PR titles are good. One main function
they serve is to help us write release notes. For this, we use the
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format.

### PR title format

```
<type>[(scope)][!]: <description>
```

- **`type`** — what kind of change (see table below).
- **`scope`** *(optional)* — the affected subpackage, e.g. `data`, `cli`.
- **`!`** *(optional)* — marks a breaking / user-facing API change.
- **`description`** — imperative tone, lowercase, no trailing period.

| type       | use for                                      | release notes? |
|------------|----------------------------------------------|----------------|
| `feat`     | new user-facing capability                   | yes            |
| `fix`      | bug fix                                      | yes            |
| `cleanup`  | removing legacy/dead code or deprecated APIs | yes            |
| `refactor` | internal restructuring, no behavior change   | no             |
| `docs`     | documentation only                           | no             |
| `tests`    | adding or changing tests                     | no             |
| `chore`    | CI, deps, packaging, repo housekeeping       | no             |

