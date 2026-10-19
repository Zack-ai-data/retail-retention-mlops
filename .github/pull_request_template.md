## What and why

<!-- What does this change, and why is it needed? One or two sentences. Link the issue or review finding if there is one. -->

## How it was checked

- [ ] `docker compose run --rm quality` passes (lint, format, strict types, tests, coverage floor). Paste the last lines:

```
<!-- e.g. 266 passed, coverage 99.4% -->
```

- [ ] New or changed behaviour has tests that would fail without the change
- [ ] If it touches the dashboard: `docker compose run --rm dashboard-tests` passes
- [ ] If it touches data processing or models: I ran it on the real data, not only the synthetic tests

## Does it change results?

<!-- A change that moves a pinned value in tests/test_warehouse_regression.py must say so here. -->

- [ ] No pinned value changed
- [ ] Pinned values changed on purpose. For each, the old value, the new value and the reason:

| Value | Before | After | Why |
|---|---|---|---|
| | | | |

- [ ] Models need retraining after this change (the registry keeps the old champion until the gate promotes a new one)

## Risk and rollback

<!-- What could go wrong, and how would you undo it? For schema changes: is it safe to run twice, and does it touch existing data? -->

## Also updated

- [ ] README / `docs/problem_statement.md` if behaviour or assumptions changed
- [ ] Lock files, if dependencies changed (see CONTRIBUTING.md)
- [ ] No secrets, `.env` contents or private review material in this change
