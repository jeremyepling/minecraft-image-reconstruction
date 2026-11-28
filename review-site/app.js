const app = document.querySelector('#app');
const query = new URLSearchParams(location.search);
const route = /^\/runs\/([a-z0-9-]+)\/(scores|review)\/?$/.exec(location.pathname);
const defaultRun = 'floating-v1-guarded-refit';
const runId = route?.[1] || query.get('run') || defaultRun;
const view = route?.[2] || (query.get('view') === 'review' ? 'review' : 'scores');
const link = page => route ? `/runs/${runId}/${page}` : `/?run=${runId}&view=${page}`;
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
  const active = () => cases[index];
  const note = () => notes[active().id] ??= {text: '', tags: [], spot: null, imageView: 'changes'};
  const annotated = value => value.text.trim() || value.tags.length || value.spot;
  const feedback = () => cases.filter(item => notes[item.id] && annotated(notes[item.id]))
    .map(item => ({caseId: item.id, title: item.title, stage: item.stage, ...notes[item.id]}));

  function noteState() {
    const count = feedback().length;
    const availability = persistent ? 'Saved on this device' : 'This tab only';
    find('note-state').textContent = count ? `${count} of ${cases.length} noted · ${availability}` : '';
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
    const layer = active().modes.find(item => item.key === mode);
    image.src = layer.url;
    image.alt = `${active().title} · ${layer.label}`;
    find('image-caption').textContent = layer.caption;
  }

  function setZoom(value) {
    zoom = value;
    plane.style.setProperty('--zoom', value ? '200%' : '100%');
    find('zoom').setAttribute('aria-pressed', String(value));
    find('zoom').textContent = value ? 'Fit' : '2×';
    if (value) {
      viewport.scrollLeft = viewport.clientWidth / 2;
      viewport.scrollTop = viewport.clientHeight / 2;
    }
  }

  function showCase(to) {
    index = to;
    const item = active();
    picker.value = index;
    field('progress', `Case ${index + 1} of ${cases.length}`);
    field('case', item.title);
    field('stage', item.stage);
    field('prompt', item.prompt);
    field('hypothesis', item.firstPass);
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
    showPin();
    image.closest('#viewport').scrollTo(0, 0);
    toggle(find('modes'), item.modes.map(layer => [layer.key, layer.label]), mode, next => { mode = next; showImage(); });
    find('tags').replaceChildren();
    for (const topic of topics) {
      const button = el('button', topic, find('tags'));
      button.type = 'button';
      button.setAttribute('aria-pressed', String(note().tags.includes(topic)));
      button.onclick = () => {
        const value = note();
        const remove = value.tags.includes(topic);
        value.tags = remove ? value.tags.filter(tag => tag !== topic) : [...value.tags, topic];
        value.imageView = mode;
        button.setAttribute('aria-pressed', String(!remove));
        save();
      };
    }
    textarea.value = note().text;
    find('prev').disabled = index === 0;
    find('next').disabled = index === cases.length - 1;
    history.replaceState(null, '', `#${item.id}`);
    noteState();
  }

  function go(to) {
    showCase(to);
    window.scrollTo(0, 0);
  }

  async function copy() {
    const text = feedback().map(item => {
      const parts = [`${item.title} · ${item.stage} [${item.caseId}]`, ...item.tags];
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
  textarea.oninput = () => { note().text = textarea.value; note().imageView = mode; save(); };
  image.onclick = event => {
    if (!marking) return;
    const bounds = image.getBoundingClientRect();
    note().spot = {x: (event.clientX - bounds.left) / bounds.width,
      y: (event.clientY - bounds.top) / bounds.height, mode};
    note().imageView = mode;
    showPin();
    setMarking(false);
    save();
  };
  const fromHash = () => Math.max(0, cases.findIndex(item => '#' + item.id === location.hash));
  window.addEventListener('hashchange', () => go(fromHash()));
  showCase(fromHash());
}

async function main() {
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(runId)) throw new Error('Unknown run.');
  for (const page of ['scores', 'review']) {
    const anchor = document.querySelector(`#${page}-link`);
    anchor.href = link(page);
    if (view === page) anchor.setAttribute('aria-current', 'page');
  }
  const response = await fetch(`/runs/${runId}/manifest.json`);
  if (!response.ok) throw new Error('This run could not be loaded.');
  const run = await response.json();
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
