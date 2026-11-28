---
name: floating-mobile-review
description: Publish and verify Floating scorecards and per-surface image reviews, and reconnect private phone annotations to measured outcomes.
---

Use the checkout's benchmark/MOBILE_REVIEW.md for both single-image reviews and scored comparisons. Use benchmark/publish_floating_review.py for matched fixture comparisons. For a source-only image, reuse the saved handoff and mobile surface evidence code; leave reference overlays and accuracy unmeasured. Give both Scores and Image review links and a case/surface deep link where useful. Keep source, mask, attempted grid, selected assembly and any reference registered to the same native view.

Use scripts for generation, ingestion and verification. Sol high handles a changed reporting path. Check score/review routes, surface switching, an empty human queue, and one phone-sized interaction when altering the interface. Ingest shared exports using benchmark.floating_review_intake. Keep private feedback on the local machine and publish only approved question text and safe aggregate statuses. Keep preference, coverage, alignment and block size separate. After a replay, attach its measured outcome to the question.

Use authenticated MCP/API or the existing Vercel CLI, preferring CLI-initiated device approval if needed. Deploy to the existing floating-review project. For supplied imagery without public release approval, use a Vercel-authenticated preview when upload is authorized. Verify both routes and the intended access mode. A local HTML file is a local fallback, and the delivery status must say whether the hosted links were verified.

Put the Perfect masks and mask-plus-plane replay effects on Scores when measured, with direct links to the corresponding mask maps. For a surface question Jeremy needs to answer, open the affected source image and include the saved mask, the perfect surface map, their aligned comparison, and the proposed interpretation. Distinguish recorded surface boundaries from the gold block grid. Say when the first pass leaves no screenshots needing his input.
