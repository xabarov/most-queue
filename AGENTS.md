# Most-Queue agent instructions

## Paid GPU experiments

Before Runpod work, read `docs/runpod_operations.md` and the current campaign
policy under `configs/runpod/`. These instructions implement the owner's
request of 2026-11-14: bounded autonomy with active cost control.

- A campaign authorization covers its routine preparation, starts, checks,
  experiments, result export, cleanup and one justified correction/retry within
  the same scope, aggregate budget and time limit. Do not ask again per command.
- Ask once when a campaign has no spending authorization, or when scope, total
  budget, hourly ceiling, GPU count, retained storage or cleanup risk expands.
  A proposed policy or an available API key is not spending authorization.
- Prepare code, image references, model revisions, data and tests locally before
  allocating a GPU. Runpod is for the measurements that require a real GPU.
- Use the exact account/key source selected by the user. Read only the required
  key into process memory; never source a whole `.env`, print credentials, commit
  them, send them to a pod, or copy another project's `.env`.
- Read current price, stock and CUDA compatibility before allocation. Include
  storage, preparation, every attempt and cleanup reserve in the cost estimate.
  Run `tools/runpod_budget.py` with a fresh plan and the persistent campaign
  ledger. Its output is a budget check, not proof that cleanup is installed.
- No paid create until the controller and detached cleanup supervisor are
  implemented and tested. Persist the unique run intent before create, the exact
  returned pod ID immediately afterward, and verify the supervisor before
  waiting for readiness or uploading anything. On supervisor failure, terminate
  the exact newly created pod immediately through the controller's fallback.
- Use one owned pod at a time. Changes of GPU, region or retry never reset the
  campaign's spend/time counters. Reconcile an ambiguous create before retrying.
- Cleanup uses exact recorded pod ID and expected name. Do not mutate neighboring
  projects' resources or perform prefix-based/account-wide cleanup.
- Completion, Stop, controller failure, the deadline, exhausted budget, or a
  stalled stage triggers export within its bounded reserve and termination.
  End a pod before an open-ended discussion or waiting for user input. Low GPU
  utilization alone is not idle during a deliberately sparse arrival experiment.
- Confirm termination through a fresh provider read. Stopping a process, closing
  SSH, `EXITED`, or one successful delete response is not the final cleanup proof.
  Never report unconfirmed cleanup as success. Retained storage bills separately.
- A local watchdog survives the agent/controller exiting, but does not guarantee
  cleanup after loss of the entire host/network or a provider outage. Do not claim
  provider-enforced TTL or a hard financial cap without verified provider support.
  Before the first paid run, establish an independent cleanup path or explicitly
  agree the supervised local-watchdog fallback once for the campaign.
- Preserve failed runs and use new held-out runs when changing a model. Read
  `docs/DOD.md` for the repository's normal verification requirements.
