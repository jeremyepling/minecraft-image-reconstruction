# Floating mobile review

Use the existing Floating Review Vercel project. The [review app repository](https://github.com/jeremyepling/minecraft-reconstruction-review) owns both frontends. Clone it beside the reconstruction checkout or pass its path with `--app-root`. This checkout owns evidence generation, verification and the archived public experiment manifests. Give each scored run a stable ID and two links.

| Link | Contents |
| --- | --- |
| `/runs/<id>/scores` | Baseline and Candidate scores, unresolved orientations, measured pipeline handoffs and matched Perfect Stage diagnostics. |
| `/runs/<id>/review` | Selected source images with aligned placement changes, gold visible boundaries and surface outlines where available. Use a case hash to link directly to a question. |

The sample uses the archived 144-view Visible Builds V1 comparison. The mobile manifest contains aggregate results and four reviewed case IDs. Annotating changes only the browser's local drafts. Copy puts case IDs and comments into plain text for chat. Share provides JSON with the comparison digest, image mode, note topics and any marked source coordinate. The app uses the phone keyboard for dictation.

In **Image review**, use **Next mask** or the mask picker and **Evidence layer** inside the Source image card. The picker compares the mask, gold reference when available, depth-partition masks and distinct grids in the same native source-image coordinates. Both review paths open with the dark gray theme and blue accent. Identical grids list all the stages that used them. The assembly receipts mark selected and unselected charts. Roof and underside charts created inside assembly are labeled as replays from the matching chart inputs, checked against the saved selection and oriented-chart count. A surface with no saved overlay shows that status explicitly. Surface notes include the ID and active grid attempt.

## Review one run

Use `floating-single-run-review` with `floating_single_run_reviewer` (Sol high) for a new screenshot or an existing output the user wants to inspect. Run the requested version and settings once, or verify the saved receipts. Pass the source hash, revision, native masks, attempted grids, selected assembly and decoded pack to `floating_review_publisher`. The reviewer owns the first failure trace and the publisher owns the hosted links. One person can fill both assignments.

Use the same evidence pipeline as the scored review. Reuse `benchmark.mobile_surface_evidence` for native outlines and chart coordinates, and `benchmark.mask_blocks.decoded_pack` for export readback. For a source-only image, the Scores page shows observed stage counts and decoded exports. Show accuracy as unmeasured and omit the gold reference layer. A fixture comparison can use the ordinary paired publisher below.

```powershell
python -m benchmark.build_floating_single_review --source $image --baseline $build --revision $revision --app-root $reviewApp --out $singleBundle
vercel deploy $singleBundle --environment preview --project floating-review
```

Pass `--candidate` to show a second export from the same image. The preview bundle contains the resized source, aligned evidence and observed counts. It does not need the full fixture or its reference labels. Its routes are `/review`, `/scores`, and `/review?surface=<mask-id>`.

The completion check for a requested Vercel review is a verified **Image review** link and **Scores** link, plus a direct surface link when there is a traced failure. For supplied imagery approved for private upload, use a protected preview on the existing project and verify its sign-in requirement. Record a concrete blocker if hosting needs authorization or fails. Keep local evidence available for inspection while the hosted link is pending.

## Publish a scored comparison

### Coordinate a round

Start from main. Run the stage report and publisher on the measured Baseline and Candidate. The output manifest contains `analysis.actions`, the matched Perfect groups, and the chooser pilot. With an active round the coordinator assigns the first discriminating machine test, checks its result, records the outcome, and repeats while the requested round has scope. The publisher itself runs deterministically and starts no background agent. One paired publication per Candidate preserves its own changes and question history. Ordinary rounds use the 100 Floating Front full-block noon views. Run lighting checks on matched selections and score decorative noon separately.

| Assignment | Skill and worker | Deliverable |
| --- | --- | --- |
| Prioritize the round | `minecraft-algorithm-benchmark` · `floating_review_coordinator`, Astra high | One ranked investment and up to three unresolved decisions |
| Compare to Perfect | `floating-perfect-stage` · `floating_score_review`, Sol high | Matched handoffs, possible gains, missing or harmful interventions |
| Find the first failure | `floating-failure-trace` · `floating_failure_analyst`, Astra high | A measured loss traced to a coordinate and a surface, where evidence permits |
| Brainstorm and test a repair | `floating-repair-design` · Astra high designs, `floating_repair_experiment`, Astra medium, implements | Two competing explanations, a distinguishing test and a family-held-out result |
| Publish and ingest | `floating-mobile-review` · `floating_review_publisher`, Sol high for verification | Two verified mobile routes and private feedback tied to its question |

On branches that lack the named roles, use the standard worker with that model, reasoning effort and skill from the main checkout. Announce the actual worker/model assignments. The report distinguishes measured downstream effects from hypotheses. Use the Perfect gain for a specific intervention only. A scorer-assisted choice across two whole packs answers a different question from a choice among attempted grids on a wall.

### Produce both links

**Surface masks and placement impact** on Scores shows the matched perfect-mask and mask-plus-plane replays. Click a case to open the native mask comparison. In Image review, **Perfect surfaces** shows the recorded frontmost surfaces, and **Mask comparison** overlays the saved Finder boundaries and missed or extra foreground. Tap **Surfaces** for a specific saved mask, its depth partitions and grid attempts, and **Mask over perfect surfaces** to inspect it against the recorded map. The gold block grid has its own layer. Whole-view coverage counts use the union of the saved masks. Per-surface overlap matches a proposal to the recorded surface with its highest intersection over union. The per-surface check can expose merges that whole-view coverage misses.

Run the command from the repository root. Supply the existing authenticated CLI with `--vercel` if it is outside PATH. Without `--publish` the command prepares a local manifest and assets under `--out`. With it, generated public imagery, surface evidence and the run manifest are committed on the current Floating branch. The command pins the GitHub asset revision and checks remote file hashes. It stages the frontend from the review app checkout and the tracked manifests from `benchmark/reviews/mobile-review-runs`, then deploys the bundle and checks both routes.

```powershell
python -m benchmark.publish_floating_review --fixture $fixture --variant original --baseline-evidence-root $baselineOriginal --evidence-root $candidateOriginal --baseline-name 'Current method' --candidate-name 'Proposed method' --carry-forward $decision --run $runId --title $title --out "runs/mobile/$runId" --app-root $reviewApp --publish
```

Each evidence directory contains one subdirectory per fixture view. Pass the `original` directory from each Expanded Split-Border replay. The publisher verifies the saved scores against the decoded packs and reads the saved source handoff, input charts, assembly charts and selection for surface evidence. The fixture bridge also reads prior local-grid replay receipts and resolves the pinned Front 100 full-block or active historical Combined Visible selection. It uses the fixture's source variant and visible labels for the overlays. A single selected frame verifies that frame. Use `--case <id>` for an additional image. Unsupported views and unavailable grid decisions should remain visible as evidence gaps.

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
4. Stage the review app with the saved manifests and deploy the staged directory to Vercel. The site is static and has no build or install step. Browser image requests go directly to the pinned public GitHub files. Keep original captures, labels, packs and private feedback outside the deployment.

Example from the repository root, with the local input paths set for the run:

```powershell
$id = 'floating-v1-guarded-refit'
$assets = "benchmark/reviews/mobile-review-assets/$id"
python -m benchmark.floating_review_app --app-root $reviewApp --out $staged
python -m benchmark.build_floating_mobile_review $stageReport $reviewQueue $frameData --assets $assets --asset-base $assetUrl --run $id --title 'Guarded depth-first refit' --site $staged
vercel deploy $staged --prod --project floating-review
```

Check `vercel whoami` before publishing. A signed-in CLI can update the shared project without the dashboard. When the CLI needs authorization, `vercel login` gives the owner a one-time link they can approve on their phone. A new, unlinked demo can use `vercel deploy --temporary` while signed out. Treat its URL as short-lived and keep the project's claim link private.

For a local check, serve the staged directory over HTTP and open `/?run=floating-v1-guarded-refit&view=review`.
