# Validation scope

The extracted CPU implementation is checked on macOS ARM64 with Python 3.10.
Regression tests compare all four decentralized methods with source commit
`d412372` on the bundled two-step input, using zero and two local workers.
They compare objectives, all x/y/z/r decisions and logical-message counts, verify
feasibility and confirm that PG keeps its initial local service fixed. Both
centralized references are solved with HiGHS and compared with the original source.

Additional tests exercise topology, selfish gain floors, greedy allocation,
communication accounting, method selection and protection of existing outputs.
Optional Gurobi cases skip when a usable solver/license is absent.

This is a code-extraction regression check, not a repeat of the 486-input DGX
benchmark. No new runtime ranking or statistical performance claim follows from
it. The retained CUDA path has not been executed on this Mac; it requires separate
validation on the DGX hardware. The tiny example is a demonstration input, not a
replacement for the experimental dataset.

Final local check (2026-11-06): **38 tests passed, 12 optional Gurobi cases skipped**.
Ruff reported no errors. An independent extraction review found no outstanding
issues after removal of the obsolete centralized runner.
