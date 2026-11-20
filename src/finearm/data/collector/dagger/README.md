# DAgger collector — design notes (stub)

Status: interface stub only. The seam it plugs into is complete.

## What DAgger needs, and where it already exists

| Requirement | Existing building block |
|---|---|
| Policy drives the follower | daemon `policy` mode + `submit_policy_action` (1 kHz, staleness hold) |
| Where actions come from | `finearm.data.collector.base.PolicySource` — satisfied by `finearm.infer` executors (RTC or stop-and-go) or `finearm.rl.serving` |
| Human correction | leader takeover: freedrive mode switch + teleop coupling (daemon modes); same-tick leader/follower pairing via `sample_coordinator` |
| Recording | the same lerobot-record bridge as teleop (`finearm.data.collector.teleop`) |
| Registry | directory scan — this directory already registers `dagger` |

## Intended flow

1. Start daemon (follower live; leader optional).
2. Loop: PolicySource.infer(obs) → submit_policy_action; operator may grab the
   leader at any time (correction phase); every tick records BOTH the policy
   action and the actually-executed (possibly human-corrected) action.
3. Dataset schema: same features as teleop plus the policy action columns.

## Non-goals

- No new daemon capabilities (policy mode and freedrive are enough).
- No trainer-side changes: the dataset is consumed by the existing
  fine-tuning pipeline.
