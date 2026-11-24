# Corrected process-verified scorer

Fixes the error in the released strict scorer described in the E1 report: on problems
that solve for an unknown, the parser read the prose after the final equation
("w = 40, so we have ...") as part of the expression, so letters inside English words
became fake inputs and correct solutions were rejected.

The fix changes one thing in `utils/solution_dependency_graph.py`: an assignment's
right-hand side is read only while it consists of numbers, operators, brackets, `=`,
spaces and standalone letters (a letter with no other letter next to it).

## Files

- `corrected_scorer.patch` - the change as a diff against the released code repo
- `solution_dependency_graph.py` - the full corrected file (drop-in replacement for `utils/solution_dependency_graph.py`)

## Apply

```bash
cd Interplay-LM-Reasoning            # a copy of the released repo
git apply ../interplay_experiments/scorer_fix/corrected_scorer.patch
```

## Validation

Renaming test (rename every variable letter in each reference solution consistently; a
correct scorer must still accept it), plus corrupted references that must be rejected:

| Check | Original scorer | Corrected scorer |
|---|---|---|
| Composition val, forward problems, renamed (1,968) | 1,968 accepted | 1,968 accepted |
| Composition val, solve-for-x problems, renamed (1,832) | 817 accepted | 1,832 accepted |
| E2 teachers val files, solve-for-x, renamed (244) | 116 accepted | 244 accepted |
| Corrupted references (wrong value / missing step / wrong answer, 9,873) | - | 0 accepted |
