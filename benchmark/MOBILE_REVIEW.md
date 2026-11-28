# Floating mobile review

Use one static Vercel project rooted at `review-site`. Give each run a stable ID and two links.

| Link | Contents |
| --- | --- |
| `/runs/<id>/scores` | Baseline and Candidate scores, unresolved orientations, measured pipeline handoffs and matched Perfect Stage diagnostics. |
| `/runs/<id>/review` | Selected source images with aligned placement changes, gold visible boundaries and surface outlines where available. Use a case hash to link directly to a question. |

The sample uses the archived 144-view Visible Builds V1 comparison. The mobile manifest contains aggregate results and four reviewed case IDs. Annotating changes only the browser's local drafts. Copy puts case IDs and comments into plain text for chat. Share provides JSON with the comparison digest, image mode, note topics and any marked source coordinate. The app uses the phone keyboard for dictation.

In **Image review**, tap **Surfaces** to step through the saved Surface Finder masks. The second picker compares the mask, gold reference, depth-partition masks and distinct grids in the same native source-image coordinates. Identical grids list all the stages that used them. The assembly receipts mark selected and unselected charts. Roof and underside charts created inside assembly are labeled as replays from the matching chart inputs, checked against the saved selection and oriented-chart count. A surface with no saved overlay shows that status explicitly. Surface notes include the ID and active grid attempt.

## Publish another run

### Coordinate a round

Start in the Floating integration checkout. Run the stage report and publisher on the measured Baseline and Candidate. The output manifest contains `analysis.actions`, the matched Perfect groups, and the chooser pilot. With an active round the coordinator assigns the first discriminating machine test, checks its result, records the outcome, and repeats while the requested round has scope. The publisher itself runs deterministically and starts no background agent. One paired publication per Candidate preserves its own changes and question history. Ordinary rounds use the 200 active Visible Builds in Combined V1. Request Decorative separately.

| Assignment | Skill and worker | Deliverable |
| --- | --- | --- |
| Prioritize the round | `minecraft-algorithm-benchmark` · `floating_review_coordinator`, Astra high | One ranked investment and up to three unresolved decisions |
| Compare to Perfect | `floating-perfect-stage` · `floating_score_review`, Sol high | Matched handoffs, possible gains, missing or harmful interventions |
| Find the first failure | `floating-failure-trace` · `floating_failure_analyst`, Astra high | A measured loss traced to a coordinate and a surface, where evidence permits |
| Brainstorm and test a repair | `floating-repair-design` · Astra high designs, `floating_repair_experiment`, Astra medium, implements | Two competing explanations, a distinguishing test and a family-held-out result |
| Publish and ingest | `floating-mobile-review` · `floating_review_publisher`, Sol high for verification | Two verified mobile routes and private feedback tied to its question |

On branches that lack the named roles, use the standard worker with that model, reasoning effort and skill from the integration checkout. Announce the actual worker/model assignments. The report distinguishes measured downstream effects from hypotheses. Use the Perfect gain for a specific intervention only. A scorer-assisted choice across two whole packs answers a different question from a choice among attempted grids on a wall.

### Produce both links

**Surface masks and placement impact** on Scores shows the matched perfect-mask and mask-plus-plane replays. Click a case to open the native mask comparison. In Image review, **Perfect surfaces** shows the recorded frontmost surfaces, and **Mask comparison** overlays the saved Finder boundaries and missed or extra foreground. Tap **Surfaces** for a specific saved mask, its depth partitions and grid attempts, and **Mask over perfect surfaces** to inspect it against the recorded map. The gold block grid has its own layer. Whole-view coverage counts use the union of the saved masks. Per-surface overlap matches a proposal to the recorded surface with its highest intersection over union. The per-surface check can expose merges that whole-view coverage misses.

Run the command from the repository root. Supply the existing authenticated CLI with `--vercel` if it is outside PATH. Without `--publish` the command prepares a local manifest and assets under `--out`. With it, only generated public imagery, surface evidence, and the manifest are committed on the current Floating branch. The command pins the GitHub asset revision, checks remote file hashes, deploys the tracked site to the existing Vercel project, and checks the live manifest and both routes.

```powershell
python -m benchmark.publish_floating_review --fixture $fixture --variant sky --baseline-evidence-root $baselineSky --evidence-root $candidateSky --baseline-name 'Current method' --candidate-name 'Proposed method' --carry-forward $decision --run $runId --title $title --out "runs/mobile/$runId" --publish
```

Each evidence directory contains one subdirectory per fixture view. The normal local-grid and V1 runners produce portable surface packages as part of the run. The fixture bridge filters to active Combined Visible views, checks the actual scored packs, and uses the fixture's source variant and visible labels for the overlays. A one-frame fixture is a one-frame verification even when 200 views are available. Use `--case <id>` for an additional image. Unsupported views and unavailable grid decisions should remain visible as evidence gaps.

For a saved V1 or stage comparison, supply `--report <report.json> --queue <queue-dir> --frames <data> --evidence-root <paired-evidence>` in place of the fixture and two variant directories. When starting with raw conditions, supply `--conditions`, `--run-root`, `--frames`, `--cohort-label`, and `--visible-available`. Supply matched `--perfect` and `--perfect-inputs` archives when generating that stage report. Review the reported scope before interpreting a gain.

### Close the feedback loop

Questions are optional. Pass `--questions <json-array>` when the source image leaves a meaningful choice after the first pass. A question includes a stable `id`, `type` (visible-ambiguity or preference), `theme`, `stage`, `caseId`, `sourceSha256`, `prompt`, `cue`, `firstPass`, `checked`, and at least two `choices` with an `answer` and the different `experiment` it would trigger. Use publishable question text. Register the visual case in the queue. Later reports reuse eligible unanswered questions when the case and source match. The team works through machine findings even with an empty human queue.

On the phone, open a question, choose the interpretation, record coverage, alignment and size separately, then share the JSON. A phone draft stays on that device until shared. The next build imports matching `<run-id>-feedback*.json` downloads on this PC. To import an explicitly shared file:

```powershell
python -m benchmark.floating_review_intake ingest $manifest --file $feedback
python -m benchmark.floating_review_intake list
python -m benchmark.floating_review_intake outcome $feedbackId --experiment $test --result $finding --changed-experiment yes --score-manifest $laterManifest
```

Private notes and measurements stay in SQLite under the repository's shared Git directory. The public manifest carries question status and aggregate usefulness measures. Attach a later paired score to distinguish answering a question from measuring an effect. For a grid chooser, use the [pilot and paired option protocol](FLOATING_GRID_CHOOSER.md). It compares the incumbent, a fixed edge rule and a learned rule on untouched building families. Exact reference grids alone cannot supply the required option outcomes.

### Check a pipeline change

Run `python -m benchmark.floating_change_impact --base <previous-revision>` on a committed change or pass specific edited paths. The report identifies affected artifacts, their consumers and unclassified inputs. Floating CI writes this plan to the job summary and exercises the review contract. Use verified stage checkpoints where supported; rerun an affected diagnostic before comparing it to a new measured candidate. Keep archived measurements pinned to their original run.

### Lower-level generation

1. Generate the ordinary stage report and visual review queue from the same comparison. Include matched Perfect Stage receipts when available and use the automatic queue as the first pass.
2. Decide which few cases need a human answer. Provide a visible cue to judge and a brief starting hypothesis. Pass optional `--labels` with each case ID mapped to `[title, stage, prompt, first pass]`.
3. Generate the manifest and optimized GitHub WebPs with `python -m benchmark.build_floating_mobile_review`. The command accepts the stage report JSON, queue directory and restored frame directory ending in `data`. For the original V1 runner, add `--run-root` pointing to the archived `runs/floating-v1` to include saved masks, depth partitions and grid attempts. Commit the imagery and surface-evidence JSON first. Set `--asset-base` to the immutable public raw GitHub URL of that commit and regenerate the manifest. Keep each previous run manifest when deploying so past links keep working.
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
