# Native proof completion: shared requirements

The user requests one proof worker per remaining theorem or identification.
Every result must describe the existing algorithms and be parameterized by the
complete algorithm, landscape, recording and calibration data it consumes.

## Mathematical contract

1. Read `docs/CLAUDE.md`, the current roadmap
   `docs/research/volume_2_remaining_proof_steps.md`, the existing proofs relevant
   to your task, and the executable configurations/update/readout code defining
   your object. Work from current results, not a generic model bearing the same
   name. Claude and Gemini MCP calls are not authorized.
2. Retain the actual full ordered update, masks, donor and history convention,
   simultaneous cloning, collision convention, both force evaluations,
   normalization, uncapped innovations, caps actually configured, geometry
   feedback actually configured, and selected-law normalization. Changing units
   is allowed; changing the kernel to obtain the desired conclusion is not.
3. Use a complete parameter record, including the algorithm/variant tag, all
   numeric and discrete configuration fields, landscape functions and their
   configured constants, boundary data, initial law, innovation/arithmetic
   convention, and consumed recording/readout/physical calibration data.
   Explicitly identify which existing complete register covers each block and
   which fields it excludes. Do not equate distinct variant registers.
4. Derive any needed profile, constant, regularity bound or sign condition from
   those inputs. A profile may be infinite or a sign condition may fail. State
   the resulting regime explicitly and describe the conclusion outside it as
   far as the proof permits. Do not replace an unproved property by a new
   assumption that the actual law has that property.
5. An exact identity valid for all well-defined configurations is useful, but
   rephrasing the desired conclusion as an equivalent unknown test is not a
   discharge. Distinguish identities, newly proved positive parameter regimes,
   actual counterexamples, inherited conditional results, and open estimates.
6. Counterexamples and obstructions must concern an actual included algorithm,
   readout, mode assignment or parameter regime. A hypothetical added physical
   interpretation is not a failure of the native algorithm. Distinguish the
   record CAR channel, scalar multiplication algebra, finite edge Hamiltonian,
   coordinate pullbacks, and a physically reconstructed transfer generator.
7. Prove only deductions supported by complete arguments. Do not claim a
   continuum Yang–Mills theorem, stationarity, nondegeneracy, uniform inequality
   or gap from a finite identity or conditional estimate. If an obligation
   cannot be discharged, retain it explicitly with the exact missing estimate.
   Try substantive lemmas and useful explicit regimes before reporting a limit.
8. Include complete formal statements and proofs with unique MyST labels. Write
   formal material only; do not create or edit Feynman prose. Cite and verify
   hypotheses of any external theorem, using primary sources when needed.

## Ownership and deliverable

You are not alone in the codebase. Other agents and the user may be editing it.
Do not revert or overwrite others' changes. Own only the proof draft explicitly
assigned to you. Do not edit shared source chapters, the roadmap, TOCs or code.
The parent will review and integrate accepted results into the main book.

Write a self-contained draft containing a task-specific parameter ledger,
substantive new formal results and full proofs, explicit regime calculations,
and a brief remaining-obligation register. Report exactly what was newly proved
and what was not, plus sources inspected and mathematical weak points for review.
