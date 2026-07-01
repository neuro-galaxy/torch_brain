# Maintaining `torch_brain`

Notes and guidelines on maintaining the `torch_brain` project.

## Commit messages

One main function of commit messages is to help us write accurate release notes.
A good commit log is essentially like a CHANGELOG. For this, we use the
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format.

**Enforcement**:
Since we mostly perform "Squash & Merge," the title of the PR becomes the main
commit message. In addition, things are set up so that the PR description forms
the commit message description. Thus, to quality-control what the
commit messages are, we have to quality-control the PR title and descriptions.

If you are reviewing a PR, please take the additional few moments to ensure
that the PR title is good and fits within our chosen format (see below). If it
is not, feel free to edit the title yourself.

### Commit message / PR title format recommendation

Full description in [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/).
Here is a TLDR. The title should look like:

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

