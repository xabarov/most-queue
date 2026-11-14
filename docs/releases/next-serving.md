# Next serving release — prepared changes

Status: draft; no new version is published by this preparation. Fresh GPU
validation of the new policy is still pending. PyPI 2.11 predates these changes.

The source checkout can now replay the saved Runpod measurements with exact
engine arrivals, separate client and server delays, and compare paired client
TTFT/TPOT SLO outcomes. `AdvanceReservationSim` adds a policy in which slots can
be reserved before compute starts and the first output token is emitted at
prefill completion. The original simulator and calibration API remain available.

Included APIs and examples:

- `most_queue.io.runpod.read_runpod_trace`, preserving failed offers and actual
  token counts with strict timestamp/profile contracts.
- Original `fit_isolated_serving` / `replay_serving_cohort`, with the completed
  A40 v1 campaign's frozen criteria and request-level results.
- `fit_advance_reservation` / `replay_advance_reservation`, with server-stage
  fitting separated from an affine client residual.
- `compare_joint_slo` and `examples/serving_reservation_slo.py`, reporting joint
  client pass fractions and false passes/failures across a fixed threshold grid.
- The deterministic fresh-validation plan, audit tools, bounded Runpod
  controller, detached cleanup supervisor, and retained raw fixtures.

On **development data**, median burst server-TTFT error drops from 41.54% to
6.63%, and maximum burst server-TPOT error drops from 28.15% to 4.69%.
Small-prompt burst client TTFT still differs by up to 33.20%. These numbers
must not be described as fresh validation of the revised policy. The original
v1 held-out data became development data when choosing this revision.

Before tagging/publishing: finish the separately authorized
[fresh run](../research/serving-reservation-v2-plan.md), preserve all gate
outcomes and cleanup evidence, assign/update release versions, build and check
sdist/wheel, test installation outside the checkout, and replace this draft's
pending statements with the actual measured outcome. Failed targets remain
explicit limitations, never retuned away on the same validation data.
