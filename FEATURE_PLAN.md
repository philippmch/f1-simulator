# FEATURE_PLAN.md

_Last updated: 2026-03-14_

## Goal
Build **real product value**: better race realism + strategy intelligence + usable frontend experience.

## Execution Rules
- Every work cycle must pick from this file first.
- Prefer **feature-level** work over micro polish.
- Ship in vertical slices (code + checks + push).
- If blocked by product decisions, ask Philipp immediately.

## Priority Roadmap

### P0 — Simulation Realism (Core)
1. Reliability model v2 (component classes, per-team failure profiles, lap/temperature stress).
2. Incident model v2 (track risk profiles, driver aggression/consistency interaction, weather severity coupling).
3. Safety car / VSC / red flag calibration against realistic per-race rates.
4. Tire crossover realism (slick/inter/wet transition windows + strong pace penalties outside window).

### P1 — Strategy Engine (High Impact)
1. Team strategy archetypes (aggressive/balanced/conservative).
2. Traffic-aware pit optimizer (undercut/overcut valuation using local race context).
3. Dynamic strategy reactions to SC/VSC/red flag and weather swings.
4. Stint planning API (planned compounds + fallback plans).

### P2 — Product UX / Frontend
1. Minimal web UI to run simulations and view outputs directly.
2. Scenario comparison dashboard (dry vs wet vs mixed side-by-side).
3. Driver/team insight cards (win%, top-N, percentiles, reliability risk).
4. Persisted run history (save/load/re-open prior simulation reports).

## Current Sprint

### Sprint A (in progress)
- [x] Dynamic reliability + incident scaling baseline
- [ ] Safety-car/VSC/red-flag calibration layer
- [ ] Tire crossover realism improvements

### Sprint B (next)
- [x] Traffic-aware pit heuristics baseline
- [ ] Team strategy archetypes
- [ ] Dynamic mid-race strategy adjustment

### Sprint C (next)
- [x] HTML report export baseline
- [ ] Actual web UI (serve + controls + scenario compare)
- [ ] Run history and report browser

## Definition of “Meaningful Improvement”
A change should satisfy at least one:
- Adds/changes core simulation behavior users can feel in outcomes.
- Adds a user-facing capability (UI/workflow/reporting) that changes how the tool is used.
- Unlocks a major next feature (foundational architecture).

## Anti-Patterns (Avoid)
- Test-only or formatting-only heartbeat cycles unless fixing breakage.
- Tiny isolated tweaks that don’t move roadmap milestones.
- Regressing into maintenance-only mode while roadmap items are open.
