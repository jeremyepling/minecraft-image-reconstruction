---
name: floating-single-run-review
description: Run or inspect one Floating reconstruction and show the image, saved surfaces, grid attempts, and export in a verified Vercel review. Use for a supplied screenshot or a saved run when Jeremy asks to see what the current build did.
---

Read [the single-run workflow](../../../benchmark/MOBILE_REVIEW.md#review-one-run) and use [the shared publisher skill](../floating-mobile-review/SKILL.md). One owner uses the requested revision and existing settings and verifies the saved handoffs. "One turn of experiments" goes to the experiment coordinator with a one-comparison budget.

1. Confirm the source and revision. For a fresh screenshot, run the ordinary single-image entry point documented in the README. For a saved run, verify its screenshot, handoff and pack receipts.
2. Follow the source masks, count and grid attempts, orientation, selected assembly and decoded export. Show the earliest failure and link directly to its surface when the evidence supports it. Include an existing comparison only when requested. Without matched reference geometry, report observed counts and mark accuracy unmeasured.
3. Package the verified evidence with `python -m benchmark.build_floating_single_review` and the separate review app checkout. Hand the bundle to the shared mobile publisher for Image review and Scores. If a Vercel link is requested, verify the hosted routes before treating it as delivered. When access or upload authorization blocks delivery, state the exact blocker promptly.
