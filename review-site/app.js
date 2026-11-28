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
const topics = ['Coverage', 'Alignment', 'Block size', 'Depth or join', 'Material', 'Looks right'];
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

function scores(run) {
  field('scope', 'Floating comparison');
  field('title', run.title);
  field('cohort', `${run.cohort} · ${run.views}/${run.available ?? run.views} views`);
  field('datasetNote', run.datasetNote);
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
    for (const grid of surface.layers) choices.push({
      ...grid, label: grid.part ? `${grid.label} · region ${grid.part.split('~p')[1]}` : grid.label,
    });
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

function review(run) {
  const storageKey = `floating-review/${run.id}/${run.comparisonSha256}`;
  let notes = {};
  let persistent = true;
  try { notes = JSON.parse(localStorage.getItem(storageKey) || '{}'); }
  catch { persistent = false; }
  const find = id => app.querySelector(`#${id}`);
  const {cases} = run;
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
  const noteKey = () => currentSurface() ? `${active().id}/${currentSurface().id}` : active().id;
  const note = () => notes[noteKey()] ??= {text: '', tags: [], spot: null, imageView: 'changes'};
  const annotated = value => value.text.trim() || value.tags.length || value.spot;
  const feedback = () => Object.entries(notes).filter(([, value]) => annotated(value)).flatMap(([key, value]) => {
    const [caseId, surfaceId] = key.split('/');
    const item = cases.find(record => record.id === caseId);
    if (!item) return [];
    return [{caseId, title: item.title, stage: surfaceId ? 'Surface masks and grids' : item.stage,
      ...(surfaceId && {surfaceId}), ...value}];
  });

  function noteState() {
    const count = feedback().length;
    const availability = persistent ? 'Saved on this device' : 'This tab only';
    const context = currentSurface() ? `${currentSurface().id} · ` : '';
    find('note-state').textContent = count ? `${context}${count} saved ${count === 1 ? 'note' : 'notes'} · ${availability}` : context.slice(0, -3);
    for (const id of ['copy', 'save-json']) find(id).disabled = !count;
  }

  function save() {
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
    history.replaceState(null, '', `#${active().id}${currentSurface() ? '/' + currentSurface().id : ''}`);
  }

  function recordContext() {
    const selection = surfaceUI.selection();
    note().imageView = selection ? selection.layer.kind || 'attempted-grid' : mode;
    if (selection) {
      note().gridAttempt = selection.layer.label;
      note().chartId = selection.layer.chart || null;
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
    for (const topic of topics) {
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
      const parts = [`${context} [${item.caseId}]`, ...item.tags];
      if (item.spot) parts.push(`At ${Math.round(item.spot.x * 100)}% across, ${Math.round(item.spot.y * 100)}% down (${item.spot.mode})`);
      if (item.text) parts.push(item.text);
      return parts.join('\n');
    }).join('\n\n');
    try {
      await navigator.clipboard.writeText(`Floating review · ${run.id}\n\n${text}`);
      notify('Copied. Paste the notes into this chat.');
    } catch { notify('Clipboard unavailable. Tap Share to export.'); }
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
    const [caseId, surfaceId] = location.hash.slice(1).split('/');
    showCase(Math.max(0, cases.findIndex(item => item.id === caseId)), surfaceId);
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
  if (view === 'review' && !run.cases.length) app.replaceChildren(el('p', 'This run has no images queued for review.'));
  else if (view === 'review') review(run);
  else scores(run);
}

main().catch(error => {
  app.replaceChildren(el('p', error.message, null, 'muted'));
  app.setAttribute('aria-busy', 'false');
});
