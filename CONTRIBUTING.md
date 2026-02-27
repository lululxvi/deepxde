# Contributing to DeepXDE

First off, thank you for taking the time to contribute! It’s people like you who make this project better. To keep things running smoothly, please follow these guidelines.

---

## Code Style & Formatting

We use **Black** as our uncompromising code formatter to keep the codebase consistent and eliminate "tabs vs. spaces" debates.

* **Format before you commit:** Ensure your code is formatted by running:
```bash
black .

```


* **Pre-commit Hooks:** (Optional but recommended) If this repo has a `.pre-commit-config.yaml`, please run `pre-commit install` to automate this process.

---

## Pull Request Guidelines

To ensure your PR is reviewed and merged quickly, please adhere to the following:

### 1. Minimal Changes (Keep it Focused)

* **One PR = One Task:** Avoid "scope creep." If you are fixing a bug and notice a typo in an unrelated file, please fix the typo in a separate, tiny PR.
* **Easier Reviews:** Smaller PRs are significantly easier to review and much less likely to introduce unintended side effects.
* **No "Drive-by" Refactoring:** Please don't reformat entire files or change variable names unless it is the specific goal of the PR.

### 2. PR Descriptions

* Briefly explain **what** changed and **why**.
* Link to any related issues (e.g., `Closes #123`).
* Include screenshots or terminal output if there are UI or CLI changes.

---

## The Review Process

* **New to the Repo?** Reviewing code is the fastest way to learn the architecture. Ask questions and point out things that are confusing.
* **Tone:** Keep reviews constructive and kind. Use "nit:" for minor stylistic suggestions that shouldn't block a merge.
* **Responsiveness:** Please try to address review comments within 48 hours.

---

## Questions?

If you’re unsure about anything, feel free to open an Issue or reach out to the team on [Slack/Discord/Teams channel].

---