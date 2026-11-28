const app = document.querySelector('#app');
const query = new URLSearchParams(location.search);
const route = /^\/runs\/([a-z0-9-]+)\/(scores|review)\/?$/.exec(location.pathname);
const defaultRun = 'floating-v1-guarded-refit';
const runId = route?.[1] || query.get('run') || defaultRun;
const view = route?.[2] || (query.get('view') === 'review' ? 'review' : 'scores');
const link = page => route ? `/runs/${runId}/${page}` : `?run=${runId}&view=${page}`;
const fmt = number => number.toLocaleString('en-US');
const percent = score => `${score.toFixed(2)}%`;
const points = value => `${value > 0 ? '+' : ''}${value.toFixed(2)} pp`;
const metrics = [
  ['geometry_strict', 'Strict', 'Strict placement F1'],
  ['geometry_facade', 'Facade', 'Facade placement F1'],
  ['material_strict', 'Strict + ID', 'Strict position + ID F1'],
  ['material_facade', 'Facade + ID', 'Facade position + ID F1'],
];
const topics = ['Covers more visible area', 'Missing visible area', 'Alignment is off', 'Block scale is wrong', 'Depth or join', 'Material', 'Looks right'];
let toastTimeout;

function el(tag, text, parent, className) {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  if (parent) parent.append(element);
  return element;
}

function field(name, value) {
  app.querySelector(`[data-field="${name}"]`).textContent = value;
}

function notify(message) {
  const status = document.querySelector('#status');
  window.clearTimeout(toastTimeout);
  status.textContent = message;
  status.hidden = false;
  toastTimeout = window.setTimeout(() => { status.hidden = true; }, 3200);
}

function toggle(parent, entries, selected, onChange) {
  parent.replaceChildren();
  for (const [value, label] of entries) {
    const button = el('button', label, parent);
    button.type = 'button';
    button.dataset.value = value;
    button.setAttribute('aria-pressed', String(value === selected));
    button.addEventListener('click', () => {
      for (const option of parent.children) option.setAttribute('aria-pressed', String(option === button));
      onChange(value);
    });
  }
}

function columns(table, methods) {
  const header = table.createTHead().insertRow();
  el('th', 'Measure', header).scope = 'col';
  for (const {name} of methods) {
    const [arm, description] = name.split(' — ');
    const th = el('th', arm, header);
    th.scope = 'col';
    if (description) el('small', description, th);
  }
  return table.createTBody();
}

function row(table, name, values) {
  const tr = table.insertRow();
  const th = el('th', name, tr);
  th.scope = 'row';
  for (const value of values) el('td', value, tr);
  return th;
}

function evidenceLink(action, id) {
  const base = link('review');
  const key = action ? (base.includes('?') ? '&' : '?') + 'action=' + encodeURIComponent(action.id) : '';
  const surface = action?.question?.surfaceId || action?.surfaceLinks?.[id];
  return base + key + '#' + encodeURIComponent(id) + (surface ? '/' + encodeURIComponent(surface) : '');
}

function decisionCard(parent, action, run) {
  const card = el('article', '', parent, 'decision-card');
  el('p', action.stage + ' · ' + action.basis, card, 'eyebrow');
  el('h3', action.title, card);
  el('p', action.evidence, card);
  el('p', action.hypothesis, card, 'caption');
  el('strong', 'Next test', card);
  el('p', action.test, card);
  if (action.question && action.disposition === 'question') {
    if (action.title !== action.question.prompt) el('p', action.question.prompt, card);
    el('p', action.question.cue, card, 'caption');
  }
  for (const id of action.caseIds) {
    const item = run.cases.find(row => row.id === id);
    if (item) el('a', item.title, card).href = evidenceLink(action, id);
  }
}

function decisions(run, target, reviewPage) {
  if (!run.analysis) return;
  target.hidden = false;
  const actions = run.analysis.actions;
  if (!reviewPage) {
    el('h2', 'Ranked next tests', target);
    if (actions.length) decisionCard(target, actions[0], run);
    if (actions.length > 1) {
      const rest = el('details', '', target);
      el('summary', 'Other ranked tests (' + (actions.length - 1) + ')', rest);
      for (const action of actions.slice(1)) decisionCard(rest, action, run);
    }
    if (!actions.length) el('p', 'No next test identified for this comparison.', target);
    return;
  }
  const human = actions.filter(a => a.disposition === 'question' && !['answered', 'tested'].includes(a.status));
  el('h2', 'Questions for you', target);
  if (!human.length) el('p', 'No human decision is pending for this comparison.', target);
  for (const action of human) decisionCard(target, action, run);
  const metrics = run.analysis.questionMetrics;
  if (metrics && (metrics.answered || metrics.tested || metrics.proposed > human.length)) {
    const history = el('details', '', target);
    el('summary', 'Question history', history);
    el('p', metrics.presented + ' shown · ' + metrics.pending + ' pending · ' +
      metrics.answered + ' answered · ' + metrics.tested + ' tested · ' + metrics.measuredEffects + ' measured outcomes', history);
    for (const [type, values] of Object.entries(metrics.byType)) {
      el('p', type + ' · ' + values.changedExperiment + ' changed the experiment out of ' + values.tested + ' tested', history);
    }
    for (const outcome of run.analysis.outcomes) el('p', outcome.actionId + ' · ' + outcome.status, history, 'caption');
  }
  const handled = actions.filter(action => !human.includes(action));
  if (handled.length) {
    const details = el('details', '', target);
    el('summary', 'First-pass findings and follow-up tests (' + handled.length + ')', details);
    for (const action of handled) decisionCard(details, action, run);
  }
  if (run.cases.length) {
    const details = el('details', '', target);
    el('summary', 'Browse saved image evidence (' + run.cases.length + ')', details);
    for (const item of run.cases) el('a', item.title, details).href = evidenceLink(null, item.id);
  }
}

function diagnosticSummary(run) {
  if (!run.analysis) return;
  const section = app.querySelector('#diagnostics');
  section.hidden = false;
  el('h2', 'Diagnostic coverage', section);
  for (const group of run.analysis.diagnosticGroups) {
    const details = el('details', '', section);
    el('summary', group.cohort + ' · ' + group.tested + ' tested', details);
    el('p', 'Replay anchor · ' + group.anchor, details, 'caption');
    for (const item of group.rows) {
      el('p', item.stage + ' · ' + item.category + ' · ' + points(item.delta.geometry_strict) + ' strict', details);
    }
  }
  const pilot = run.analysis.chooserPilot;
  if (pilot) {
    const details = el('details', '', section);
    const pack = pilot.exportChooser;
    const grids = pilot.reviewedGrids;
    el('summary', 'Chooser pilot · ' + pack.tested + ' paired exports', details);
    const before = pack.before.geometry_strict;
    const after = pack.after.geometry_strict;
    el('p', pack.switches + ' switches among the measured exports · ' +
      points(after.f1 - before.f1) + ' strict · ' + fmt(after.correct - before.correct) +
      ' additional correct positions · ' + fmt(after.extras - before.extras) + ' change in extras.', details);
    el('p', pack.criterion, details, 'caption');
    const splits = Object.entries(grids.splits).map(([split, count]) => count + ' ' + split).join(', ');
    el('p', grids.multipleGridLayers + ' of ' + grids.parents + ' reviewed surfaces have multiple saved grid layers across ' +
      grids.views + ' views' + (splits ? ' (' + splits + ').' : '.'), details);
    for (const [split, heading] of Object.entries(pilot.headingBySplit)) {
      el('p', 'Heading chooser · ' + split + ' · ' + heading.views + ' views · ' + heading.unresolved +
        ' unresolved · ' + fmt(heading.correctGain) + ' additional correct positions.', details, 'caption');
    }
    for (const gap of pilot.missing) el('p', gap, details, 'caption');
  }
  const missing = run.analysis.diagnosticsMissing;
  if (missing.length) el('p', 'Unmeasured stages · ' + missing.join(', '), section, 'caption');
}

function scores(run) {
  decisions(run, app.querySelector('#priorities'), false);
  diagnosticSummary(run);
  field('scope', 'Floating comparison');
  field('title', run.title);
  field('cohort', `${run.cohort} · ${run.views}/${run.available ?? run.views} views`);
  field('datasetNote', run.datasetNote);
  field('carryForward', run.carryForward || 'Decision for this comparison is pending.');
  const [baseline, candidate] = run.methods;
  const delta = candidate.scores.geometry_strict.f1_percent - baseline.scores.geometry_strict.f1_percent;
  field('placementDelta', `${points(delta)} strict`);
  app.querySelector('[data-field="placementDelta"]').classList.toggle('negative', delta < 0);
  field('unresolved', `Orientation unresolved: ${baseline.failures.unresolved} baseline · ${candidate.failures.unresolved} candidate. Included in scores.`);
  const scoreTable = columns(app.querySelector('#scores'), run.methods);
  for (const [key, , label] of metrics) {
    row(scoreTable, label, run.methods.map(({scores: measures}) => percent(
      measures[key.replace('material_', 'scores_')].f1_percent)));
  }
  for (const type of ['correct', 'missed', 'extras']) {
    row(scoreTable, type[0].toUpperCase() + type.slice(1), run.methods.map(({scores: measures}) => fmt(measures.geometry_strict[type])));
  }
  const stageTable = columns(app.querySelector('#stages'), run.methods);
  for (const stage of run.stages) {
    const th = row(stageTable, stage.name, stage.values.map(value =>
      value ? `${fmt(value.numerator)} / ${fmt(value.denominator)}` : 'Unmeasured'));
    el('div', stage.measure, th, 'caption');
  }

  const chart = app.querySelector('#perfect');
  const scope = app.querySelector('#perfect-scope');
  const metricMenu = app.querySelector('#metric-switch');
  if (!run.perfect.length) {
    scope.hidden = true;
    el('p', 'No matched Perfect Stage replay', chart, 'caption');
    return;
  }
  for (const [label, tested] of new Map(run.perfect.map(row => [row.cohort, row.tested]))) {
    scope.add(new Option(`${label} · ${tested}`, label));
  }
  let metric = metrics[0][0];
  const draw = () => {
    chart.replaceChildren();
    for (const item of run.perfect.filter(row => row.cohort === scope.value)) {
      const before = item.before[metric].f1;
      const after = item.after[metric].f1;
      const details = el('details', '', chart, 'perfect-row');
      const summary = el('summary', '', details);
      el('span', item.stage, summary);
      const readout = el('span', `${percent(after)} · `, summary, 'chart-readout');
      el('strong', points(after - before), readout, after < before ? 'negative' : '');
      const bar = el('div', '', details, 'bar-track');
      bar.setAttribute('aria-hidden', 'true');
      el('i', '', bar).style.width = percent(after);
      el('b', '', bar).style.left = percent(before);
      el('p', `${percent(before)} source-only → ${percent(after)} with reference`, details, 'caption');
      el('p', `${item.kind}. ${item.change}`, details, 'caption');
    }
  };
  toggle(metricMenu, metrics, metric, value => { metric = value; draw(); });
  scope.addEventListener('change', draw);
  draw();
}

function surfaceTrace(container, surface, evidence) {
  container.replaceChildren();
  if (surface.placementChange) {
    const change = surface.placementChange;
    el('p', 'Correct positions gained ' + fmt(change.gained) + ' · lost ' + fmt(change.lost) +
      ' · extras added ' + fmt(change.extrasAdded) + ' · removed ' + fmt(change.extrasRemoved), container);
  }
  for (const [arm, record] of Object.entries(surface.exports || {})) {
    const label = el('details', '', container);
    const cells = record.cells || [];
    const overlaps = record.overlaps || [];
    el('summary', arm + ' · ' + fmt(cells.length) + ' owned cells · ' + fmt(overlaps.length) + ' lost writes', label);
    el('p', 'Selected charts · ' + ((record.selectedCharts || []).join(', ') || 'none'), label);
    for (const lost of overlaps) {
      el('p', lost.xyz.join(', ') + ' · ' + lost.chart + ' (' + lost.block + ') lost ownership to ' +
        lost.winnerChart + ' (' + lost.winnerBlock + ')', label, 'caption');
    }
    if (cells.length) {
      const owned = el('details', '', label);
      el('summary', 'Decoded coordinates and block IDs', owned);
      owned.addEventListener('toggle', () => {
        if (!owned.open || owned.children.length > 1) return;
        for (const cell of cells) el('p', cell.xyz.join(', ') + ' · ' + cell.block + ' · ' + cell.chart +
          (cell.fallback ? ' · fallback material' : ''), owned, 'caption');
      });
    }
  }
  for (const decision of surface.decisions || []) {
    const details = el('details', '', container);
    const receipt = decision.detail;
    const charts = receipt?.chart_ids || [];
    el('summary', [decision.arm, decision.kind, ...charts].join(' · '), details);
    if (typeof receipt === 'string') {
      el('p', receipt, details);
    } else if (receipt) {
      const {image_edges, ...fields} = receipt;
      el('pre', JSON.stringify(fields, null, 2), details);
    }
  }
  for (const gap of evidence.gaps || []) el('p', gap.stage + ' · ' + gap.reason, container, 'caption');
}

function surfaceReview(find, changed) {
  const button = find('surface-mode');
  const controls = find('surface-controls');
  const picker = find('surface-picker');
  const layers = find('surface-layer');
  const canvas = find('surface-overlay');
  const receipt = find('surface-receipt');
  const cache = new Map();
  let item, data, surface, choices, layer;
  let enabled = false;
  let load = 0;

  function draw() {
    const part = layer.part && surface.parts[layer.part];
    const shape = new Path2D(part?.mask || surface.mask);
    const outside = new Path2D();
    outside.rect(0, 0, canvas.width, canvas.height);
    outside.addPath(shape);
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = 'rgba(9, 17, 14, .57)';
    context.fill(outside, 'evenodd');
    if (!part) {
      context.strokeStyle = '#bcf2c5';
      context.lineWidth = 3;
      context.stroke(shape);
    }
    if (layer.path) {
      const grid = new Path2D(layer.path);
      context.lineWidth = 7;
      context.strokeStyle = '#102923';
      context.stroke(grid);
      context.lineWidth = 3.8;
      context.strokeStyle = '#78eeff';
      context.stroke(grid);
    }
    if (layer.cells) {
      const owned = new Path2D();
      for (const cell of layer.cells) {
        if (!cell.quad?.length) continue;
        owned.moveTo(...cell.quad[0]);
        for (const point of cell.quad.slice(1)) owned.lineTo(...point);
        owned.closePath();
      }
      context.strokeStyle = '#78eeff';
      context.lineWidth = 2.5;
      context.stroke(owned);
    }
    if (layer.markers) {
      const colors = {gained:'#65e6e8', lost:'#ff78ae', extrasAdded:'#ffc461', extrasRemoved:'#be9cff'};
      for (const marker of layer.markers) {
        const [x, y] = marker.source;
        context.fillStyle = colors[marker.kind];
        context.strokeStyle = '#151918';
        context.lineWidth = 1.5;
        context.beginPath();
        context.arc(x, y, 4, 0, 2 * Math.PI);
        context.fill();
        context.stroke();
      }
    }
    canvas.hidden = false;
  }

  function selectLayer(index) {
    layer = choices[index];
    layers.value = index;
    const {count} = surface;
    const part = layer.part && surface.parts[layer.part];
    const detail = [];
    if (part) {
      const fits = part.fits.map(name => name === 'complete' ? 'full' : name);
      detail.push(`${fmt(part.pixels)} region pixels · ${part.inherited} inherited cells`);
      detail.push(fits.length ? `Refits saved: ${fits.join(', ')}.` : 'No fresh grid accepted for this region.');
    } else if (count) {
      const blocks = count.blocks?.map(value => value ?? '?').join(' × ');
      detail.push(blocks && count.blocks.some(value => value !== null) ? `Block Counter: ${blocks}.` : 'Block Counter found no dimensions.');
    }
    if (part?.fitEvidence) {
      for (const [kind, fit] of Object.entries(part.fitEvidence)) detail.push(kind + ' · ' + fit.reason + '.');
    }
    if (layer.cells) detail.push(fmt(layer.cells.length) + ' owned exported cells.');
    if (layer.markers) detail.push('Source-face centers · cyan correct gained · pink correct lost · amber extras added · violet extras removed.');
    if (layer.path) detail.push(`${fmt(layer.cellCount)} grid cells · ${layer.stages.join(' · ')}.`);
    if (layer.status?.length) detail.push(layer.status.join(' · '));
    if (!surface.layers.length) detail.push('No grid overlay was saved for this surface.');
    else if (layer.kind === 'mask') detail.push(`${surface.layers.length} distinct saved grid overlays.`);
    receipt.textContent = detail.join(' ');
    draw();
    changed();
  }

  function select(index) {
    surface = data.surfaces[index];
    picker.value = index;
    find('prev-surface').disabled = index === 0;
    find('next-surface').disabled = index === data.surfaces.length - 1;
    choices = [{label: 'Surface mask', kind: 'mask'}, {label: 'Gold reference', kind: 'gold'}];
    for (const part of Object.keys(surface.parts)) choices.push({label: `Depth region ${part.split('~p')[1]}`, part});
    const attempts = new Map();
    for (const grid of surface.layers) {
      const label = grid.part ? `${grid.label} · region ${grid.part.split('~p')[1]}` : grid.label;
      const identity = JSON.stringify([grid.chart, label]);
      const occurrence = attempts.get(identity) || 0;
      attempts.set(identity, occurrence + 1);
      choices.push({...grid, label, occurrence});
    }
    for (const [arm, exported] of Object.entries(surface.exports || {})) {
      choices.push({kind: 'export', label: arm + ' · exported cells', cells: exported.cells || []});
    }
    if (surface.placementChange?.markers.length) {
      choices.push({kind:'changes', label:'Traced placement changes', markers:surface.placementChange.markers});
    }
    find('surface-trace').hidden = data.schema !== 2;
    if (data.schema === 2) surfaceTrace(find('trace-body'), surface, data);
    layers.replaceChildren(...choices.map(({label}, option) => new Option(label, option)));
    selectLayer(0);
  }

  function close() {
    load += 1;
    enabled = false;
    data = surface = layer = null;
    button.setAttribute('aria-pressed', 'false');
    find('modes').hidden = false;
    controls.hidden = canvas.hidden = true;
  }

  async function open(id) {
    if (!item.surfaceEvidence) return;
    const request = ++load;
    enabled = true;
    button.setAttribute('aria-pressed', 'true');
    find('modes').hidden = true;
    controls.hidden = false;
    picker.disabled = layers.disabled = true;
    find('prev-surface').disabled = find('next-surface').disabled = true;
    receipt.textContent = 'Loading saved surface evidence…';
    try {
      const stored = cache.get(item.id);
      const evidence = stored || await fetch(item.surfaceEvidence).then(response => {
        if (!response.ok) throw new Error('Missing evidence');
        return response.json();
      });
      if (request !== load) return;
      if (evidence.id !== item.id || evidence.sourceSha256 !== item.sourceSha256 ||
          evidence.size[0] !== item.width || evidence.size[1] !== item.height || !evidence.surfaces.length) {
        throw new Error('Invalid surface data');
      }
      data = evidence;
      cache.set(item.id, evidence);
      canvas.width = item.width;
      canvas.height = item.height;
      picker.replaceChildren(...data.surfaces.map(({id}, index) => new Option(id, index)));
      picker.disabled = layers.disabled = false;
      select(Math.max(0, data.surfaces.findIndex(row => row.id === id)));
    } catch {
      if (request === load) receipt.textContent = 'Saved surface evidence could not load. Tap Surfaces to go back.';
    }
  }

  button.onclick = () => {
    if (!enabled) { void open(); return; }
    close();
    changed();
  };
  picker.onchange = () => select(Number(picker.value));
  layers.onchange = () => selectLayer(Number(layers.value));
  find('prev-surface').onclick = () => select(Number(picker.value) - 1);
  find('next-surface').onclick = () => select(Number(picker.value) + 1);
  return {
    reset(next) { close(); item = next; button.hidden = !item.surfaceEvidence; },
    open,
    enabled: () => enabled,
    selection: () => surface && {surface, layer},
  };
}

const assessmentFields = [
  ['coverage', 'Coverage', 'Does it cover the visible surface?'],
  ['alignment', 'Alignment', 'Do the grid lines align with the visible seams?'],
  ['blockSize', 'Block size', 'Is the block size correct?'],
];

function assessmentControls(container, initial, changed) {
  container.replaceChildren();
  for (const [key, title, prompt] of assessmentFields) {
    const group = el('fieldset', '', container);
    el('legend', prompt, group);
    const pick = el('select', '', group);
    pick.setAttribute('aria-label', title + ' judgment');
    for (const [value, label] of [['', 'Unanswered'], ['yes', 'Yes'], ['no', 'No'], ['unclear', 'Unclear']]) {
      pick.add(new Option(label, value));
    }
    const label = el('label', title + ' note (optional)', group);
    const input = el('textarea', '', label);
    input.rows = 1;
    pick.value = initial?.[key]?.answer || '';
    input.value = initial?.[key]?.note || '';
    const update = () => changed(key, {answer: pick.value, note: input.value});
    pick.onchange = input.oninput = update;
  }
}

function gridKey(parent, chart, label, occurrence = 0) {
  return parent + '#' + encodeURIComponent(JSON.stringify([chart || '', label, occurrence]));
}

function emptyNote() {
  return {text: '', tags: [], spot: null, imageView: 'changes'};
}

function separateGridDrafts(notes) {
  for (const [key, draft] of Object.entries(notes)) {
    if (key.includes('#') || !['attempted-grid', 'export'].includes(draft.imageView) || !draft.gridAttempt) continue;
    const attempt = gridKey(key, draft.chartId, draft.gridAttempt);
    if (notes[attempt]) continue;
    notes[attempt] = {...draft};
    if (draft.answer) {
      notes[key] = {...emptyNote(), actionId: draft.actionId, questionType: draft.questionType, answer: draft.answer};
      delete notes[attempt].answer;
    } else {
      delete notes[key];
    }
  }
}

function assessmentLines(assessments) {
  return assessmentFields.flatMap(([key, label]) => {
    const item = assessments?.[key];
    return item ? [label + ' · ' + [item.answer, item.note].filter(Boolean).join(' · ')] : [];
  });
}

function review(run) {
  const storageKey = `floating-review/${run.id}/${run.comparisonSha256}`;
  let notes = {};
  let persistent = true;
  try { notes = JSON.parse(localStorage.getItem(storageKey) || '{}'); }
  catch { persistent = false; }
  separateGridDrafts(notes);
  const find = id => app.querySelector(`#${id}`);
  const {cases} = run;
  decisions(run, find('review-queue'), true);
  const inspector = find('case-inspector');
  if (!cases.length) {
    inspector.hidden = true;
    if (!run.analysis) el('p', 'This run has no images queued for review.', app);
    return;
  }
  const picker = find('case-picker');
  const textarea = find('note');
  const image = find('evidence');
  const viewport = find('viewport');
  const plane = find('image-plane');
  const pin = find('pin');
  let index = 0;
  let mode = 'changes';
  let zoom = false;
  let marking = false;
  let previousSurface = null;
  const active = () => cases[index];
  const surfaceUI = surfaceReview(find, () => {
    showImage();
    showPrompt();
    loadNote();
    const next = currentSurface()?.id || null;
    setZoom(zoom, previousSurface !== next);
    previousSurface = next;
    markLocation();
  });
  const currentSurface = () => surfaceUI.selection()?.surface;
  const chosenAction = () => run.analysis?.actions.find(a => a.id === query.get('action') && a.caseIds.includes(active().id));
  const question = () => {
    const action = chosenAction();
    if (action?.disposition !== 'question') return null;
    const target = action.question.surfaceId;
    return !target || target === currentSurface()?.id ? action.question : null;
  };
  const parentKey = () => {
    const base = currentSurface() ? active().id + '/' + encodeURIComponent(currentSurface().id) : active().id;
    return question() ? base + '@' + encodeURIComponent(question().id) : base;
  };
  const noteKey = () => {
    const layer = surfaceUI.selection()?.layer;
    return (layer?.path || layer?.kind === 'export') ? gridKey(parentKey(), layer.chart, layer.label, layer.occurrence) : parentKey();
  };
  const note = () => notes[noteKey()] ??= emptyNote();
  const annotated = value => value.text.trim() || value.tags.length || value.spot || value.preference || value.answer || Object.keys(value.assessments || {}).length;
  const feedback = () => Object.entries(notes).filter(([, value]) => annotated(value)).flatMap(([key, value]) => {
    const [identity] = key.split('#')[0].split('@');
    const [caseId, encodedSurface] = identity.split('/');
    const surfaceId = encodedSurface && decodeURIComponent(encodedSurface);
    const item = cases.find(record => record.id === caseId);
    if (!item) return [];
    return [{caseId, sourceSha256: item.sourceSha256, title: item.title, stage: surfaceId ? 'Surface masks and grids' : item.stage,
      ...(surfaceId && {surfaceId}), ...value}];
  });

  function noteState() {
    const count = feedback().length;
    const availability = persistent ? 'Saved on this device' : 'This tab only';
    const context = currentSurface() ? `${currentSurface().id} · ` : '';
    find('note-state').textContent = count ? `${context}${count} saved ${count === 1 ? 'note' : 'notes'} · ${availability}` : context.slice(0, -3);
    for (const id of ['copy', 'save-json']) find(id).disabled = !count;
  }

  function save(draft = note()) {
    if (question()) Object.assign(draft, {actionId: chosenAction().id, questionType: question().type});
    draft.updatedAt = new Date().toISOString();
    try { localStorage.setItem(storageKey, JSON.stringify(notes)); }
    catch { persistent = false; }
    noteState();
  }

  function setMarking(value) {
    marking = value;
    plane.classList.toggle('marking', value);
    find('mark').setAttribute('aria-pressed', String(value));
    find('mark').textContent = value ? 'Tap image' : 'Mark spot';
  }

  function showPin() {
    const location = note().spot;
    pin.hidden = !location;
    find('clear-mark').hidden = !location;
    if (location) {
      pin.style.left = percent(location.x * 100);
      pin.style.top = percent(location.y * 100);
    }
  }

  function showImage() {
    const selection = surfaceUI.selection();
    const key = selection ? (selection.layer.kind === 'gold' ? 'gold' : 'source') : mode;
    const background = active().modes.find(item => item.key === key);
    if (image.getAttribute('src') !== background.url) image.src = background.url;
    image.alt = selection ? `${active().title} · ${selection.surface.id} · ${selection.layer.label}` :
      `${active().title} · ${background.label}`;
    if (!selection) {
      find('image-caption').textContent = background.caption;
      return;
    }
    const {layer, surface} = selection;
    const region = layer.part && surface.parts[layer.part];
    if (layer.kind === 'export') {
      find('image-caption').textContent = 'Cyan outlines show owned exported cells with saved image coordinates.';
      return;
    }
    find('image-caption').textContent = layer.path ? (region ? 'Cyan attempted grid · highlighted depth region' :
      'Cyan attempted grid · pale green surface outline') :
      layer.kind === 'gold' ? 'Gold recorded visible face boundaries' : 'Highlighted support in the source image';
  }

  function showPrompt() {
    const selection = surfaceUI.selection();
    let {prompt, firstPass} = active();
    if (selection) {
      const {layer} = selection;
      if (layer.path) {
        prompt = 'Does this grid follow the visible block seams?';
        firstPass = 'Compare with Gold reference. Check spacing, alignment and the visible area it covers.';
      } else if (layer.part) {
        prompt = 'Where should this depth region stop?';
        firstPass = 'Look for a return, a material edge or an opening that separates it from the next plane.';
      } else {
        prompt = 'Does the mask cover the right surface?';
        firstPass = 'Check the edges for missed visible pixels, neighboring faces and openings.';
      }
    }
    const action = chosenAction();
    if (action) {
      prompt = question()?.prompt || action.title;
      firstPass = [action.evidence, action.hypothesis, question()?.cue, 'Next test · ' + action.test].filter(Boolean).join(' ');
    } else if (run.analysis) {
      prompt = 'Image evidence';
      firstPass = active().stage;
    }
    field('prompt', prompt);
    field('hypothesis', firstPass);
  }

  function setZoom(value, recenter = false) {
    const wasZoomed = zoom;
    zoom = value;
    plane.style.setProperty('--zoom', value ? '200%' : '100%');
    find('zoom').setAttribute('aria-pressed', String(value));
    find('zoom').textContent = value ? 'Fit' : '2×';
    if (value && (!wasZoomed || recenter)) {
      const [left, top, right, bottom] = currentSurface()?.bounds || [0, 0, active().width, active().height];
      viewport.scrollTo(viewport.clientWidth * ((left + right) / active().width - .5),
        viewport.clientHeight * ((top + bottom) / active().height - .5));
    }
  }

  function markLocation() {
    history.replaceState(null, '', `#${active().id}${currentSurface() ? '/' + encodeURIComponent(currentSurface().id) : ''}`);
  }

  function recordContext() {
    const selection = surfaceUI.selection();
    note().imageView = selection ? selection.layer.kind || 'attempted-grid' : mode;
    if (selection) {
      note().gridAttempt = selection.layer.label;
      note().chartId = selection.layer.chart || null;
      if (selection.layer.path) note().gridOccurrence = selection.layer.occurrence;
    }
  }

  function showCase(to, surfaceId = null, keepSurfaces = false) {
    index = to;
    const item = active();
    surfaceUI.reset(item);
    previousSurface = null;
    picker.value = index;
    field('progress', `Case ${index + 1} of ${cases.length}`);
    field('case', item.title);
    field('stage', item.stage);
    const counts = find('case-counts');
    counts.replaceChildren();
    el('span', `+${fmt(item.counts.gain)} correct`, counts, 'gain');
    el('span', `−${fmt(item.counts.loss)} correct`, counts, 'loss');
    el('span', `Extras +${fmt(item.counts.add)} / −${fmt(item.counts.remove)}`, counts, 'caption');
    mode = item.modes.some(layer => layer.key === note().imageView) ? note().imageView : item.modes[0].key;
    viewport.style.setProperty('--aspect', `${item.width} / ${item.height}`);
    image.width = item.width;
    image.height = item.height;
    setZoom(false);
    setMarking(false);
    showImage();
    showPrompt();
    image.closest('#viewport').scrollTo(0, 0);
    toggle(find('modes'), item.modes.map(layer => [layer.key, layer.label]), mode, next => { mode = next; showImage(); });
    find('prev').disabled = index === 0;
    find('next').disabled = index === cases.length - 1;
    markLocation();
    loadNote();
    if (surfaceId || keepSurfaces) void surfaceUI.open(surfaceId);
  }

  function loadNote() {
    showPin();
    find('tags').replaceChildren();
    for (const topic of new Set([...topics, ...note().tags])) {
      const button = el('button', topic, find('tags'));
      button.type = 'button';
      button.setAttribute('aria-pressed', String(note().tags.includes(topic)));
      button.onclick = () => {
        const value = note();
        const remove = value.tags.includes(topic);
        value.tags = remove ? value.tags.filter(tag => tag !== topic) : [...value.tags, topic];
        recordContext();
        button.setAttribute('aria-pressed', String(!remove));
        save();
      };
    }
    const choices = find('question-choices');
    choices.replaceChildren();
    if (question()) {
      const selected = question();
      const decision = notes[parentKey()] ??= emptyNote();
      toggle(choices, selected.choices.map(c => [c.answer, c.answer]), decision.answer, answer => {
        decision.answer = answer;
        save(decision);
      });
    }
    const preference = find('arm-preference');
    preference.value = note().preference || '';
    preference.onchange = () => { note().preference = preference.value; recordContext(); save(); };
    assessmentControls(find('criterion-fields'), note().assessments, (key, value) => {
      const assessments = note().assessments ??= {};
      if (value.answer || value.note.trim()) assessments[key] = value;
      else delete assessments[key];
      recordContext();
      save();
    });
    textarea.value = note().text;
    noteState();
  }

  function go(to) {
    showCase(to, null, surfaceUI.enabled());
    window.scrollTo(0, 0);
  }

  async function copy() {
    const text = feedback().map(item => {
      const context = [item.title, item.stage, item.surfaceId, item.gridAttempt].filter(Boolean).join(' · ');
      const parts = [`${context} [${item.caseId}]`, item.answer, item.preference && ('Preference · ' + item.preference), ...item.tags].filter(Boolean);
      parts.push(...assessmentLines(item.assessments));
      if (item.spot) parts.push(`At ${Math.round(item.spot.x * 100)}% across, ${Math.round(item.spot.y * 100)}% down (${item.spot.mode})`);
      if (item.text) parts.push(item.text);
      return parts.join('\n');
    }).join('\n\n');
    try {
      await navigator.clipboard.writeText(`Floating review · ${run.id}\n\n${text}`);
      notify('Copied. Paste the notes into this chat.');
    } catch { notify('Clipboard unavailable. Tap Export notes to save the file.'); }
  }

  async function share() {
    const data = {schema: 1, runId: run.id, comparisonSha256: run.comparisonSha256, feedback: feedback()};
    const file = new File([JSON.stringify(data, null, 2) + '\n'], `${run.id}-feedback.json`, {type: 'application/json'});
    if (navigator.canShare?.({files: [file]})) {
      try { await navigator.share({files: [file]}); return; }
      catch (error) { if (error.name === 'AbortError') return; }
    }
    const url = URL.createObjectURL(file);
    const a = document.createElement('a');
    a.href = url;
    a.download = file.name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  cases.forEach(({title}, i) => picker.add(new Option(title, i)));
  picker.onchange = () => go(Number(picker.value));
  find('next').onclick = () => go(index + 1);
  find('prev').onclick = () => go(index - 1);
  find('zoom').onclick = () => setZoom(!zoom);
  find('mark').onclick = () => setMarking(!marking);
  find('clear-mark').onclick = () => { note().spot = null; showPin(); save(); };
  find('copy').onclick = copy;
  find('save-json').onclick = share;
  textarea.oninput = () => { note().text = textarea.value; recordContext(); save(); };
  image.onclick = event => {
    if (!marking) return;
    const bounds = image.getBoundingClientRect();
    recordContext();
    note().spot = {x: (event.clientX - bounds.left) / bounds.width,
      y: (event.clientY - bounds.top) / bounds.height, mode: note().gridAttempt || mode};
    showPin();
    setMarking(false);
    save();
  };
  function fromHash() {
    const [caseId, encodedSurface] = location.hash.slice(1).split('/');
    let surfaceId;
    try { surfaceId = encodedSurface && decodeURIComponent(encodedSurface); } catch { surfaceId = null; }
    const selected = cases.findIndex(item => item.id === caseId);
    inspector.hidden = Boolean(run.analysis && selected < 0);
    if (!inspector.hidden) showCase(Math.max(0, selected), surfaceId);
  }
  window.addEventListener('hashchange', fromHash);
  fromHash();
}

async function main() {
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(runId)) throw new Error('Unknown run.');
  for (const page of ['scores', 'review']) {
    const anchor = document.querySelector(`#${page}-link`);
    anchor.href = link(page);
    if (view === page) anchor.setAttribute('aria-current', 'page');
  }
  const embedded = document.querySelector('#embedded-run');
  const response = embedded ?? await fetch(`/runs/${runId}/manifest.json`);
  if (!embedded && !response.ok) throw new Error('This run could not be loaded.');
  const run = embedded ? JSON.parse(embedded.textContent) : await response.json();
  if (run.schema !== 1 || run.id !== runId) throw new Error('The run manifest is incompatible.');
  document.title = `${view === 'scores' ? 'Scores' : 'Image review'} · ${run.title} · Floating`;
  app.replaceChildren(document.querySelector(`#${view}-view`).content.cloneNode(true));
  app.setAttribute('aria-busy', 'false');
  if (view === 'review') review(run);
  else scores(run);
}

main().catch(error => {
  app.replaceChildren(el('p', error.message, null, 'muted'));
  app.setAttribute('aria-busy', 'false');
});
