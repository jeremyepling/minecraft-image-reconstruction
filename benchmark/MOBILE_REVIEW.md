# Floating mobile review

Use one static Vercel project rooted at `review-site`. Give each run a stable ID and two links.

| Link | Contents |
| --- | --- |
| `/runs/<id>/scores` | Baseline and Candidate scores, unresolved orientations, measured pipeline handoffs and matched Perfect Stage diagnostics. |
| `/runs/<id>/review` | Selected source images with aligned placement changes, gold visible boundaries and surface outlines where available. Use a case hash to link directly to a question. |

The sample uses the archived 144-view Visible Builds V1 comparison. The mobile manifest contains aggregate results and four reviewed case IDs. Annotating changes only the browser's local drafts. Copy puts case IDs and comments into plain text for chat. Share provides JSON with the comparison digest, image mode, note topics and any marked source coordinate. The app uses the phone keyboard for dictation.

## Publish another run

1. Generate the ordinary stage report and visual review queue from the same comparison. Include matched Perfect Stage receipts when available and use the automatic queue as the first pass.
2. Decide which few cases need a human answer. Provide a visible cue to judge and a brief starting hypothesis. Pass optional `--labels` with each case ID mapped to `[title, stage, prompt, first pass]`.
3. Generate the manifest and optimized GitHub WebPs with `python -m benchmark.build_floating_mobile_review`. The command accepts the stage report JSON, queue directory and restored frame directory ending in `data`. Commit the imagery first. Set `--asset-base` to the immutable public raw GitHub URL containing that imagery commit and regenerate the manifest. Keep all previous `review-site/runs/<id>/manifest.json` files when deploying so past links keep working.
4. Deploy just `review-site` to the same Vercel project. The site is static and has no build or install step. Browser image requests go directly to the pinned public GitHub files. Keep original captures, labels, packs and private feedback outside the deployment.

Example from the repository root, with the local input paths set for the run:

```powershell
$id = 'floating-v1-guarded-refit'
$assets = "benchmark/reviews/mobile-review-assets/$id"
python -m benchmark.build_floating_mobile_review $stageReport $reviewQueue $frameData --assets $assets --asset-base $assetUrl --run $id --title 'Guarded depth-first refit'
vercel deploy review-site --prod --project floating-review
```

Check `vercel whoami` before publishing. A signed-in CLI can update the shared project without the dashboard. When the CLI needs authorization, `vercel login` gives the owner a one-time link they can approve on their phone. A new, unlinked demo can use `vercel deploy --temporary` while signed out. Treat its URL as short-lived and keep the project's claim link private.

For a local check, serve `review-site` over HTTP and open `/?run=floating-v1-guarded-refit&view=review`.
