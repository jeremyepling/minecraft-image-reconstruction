import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {runInNewContext} from 'node:vm';

const site = new URL('../review-site/', import.meta.url);
const run = 'floating-v1-guarded-refit';
const markup = readFileSync(new URL('index.html', site), 'utf8');
const script = readFileSync(new URL('app.js', site), 'utf8');
const data = JSON.parse(readFileSync(new URL(`runs/${run}/manifest.json`, site), 'utf8'));
const definitions = script.slice(0, script.indexOf('\nmain().catch'));

for (const view of ['scores', 'review']) {
  test(`the ${view} route selects its real HTML template`, async () => {
    const rendered = new Error('Route initialized and template selected');
    const app = {replaceChildren() { throw rendered; }};
    const document = {
      querySelector(selector) {
        if (selector === '#app') return app;
        if (selector.endsWith('-link')) return {setAttribute() {}};
        const templateId = selector.slice(1);
        if (markup.includes(`<template id="${templateId}">`)) {
          return {content: {cloneNode: () => templateId}};
        }
        return null;
      },
    };
    const main = runInNewContext(definitions + '\nmain()', {
      document,
      location: new URL(`https://example.test/runs/${run}/${view}`),
      URLSearchParams,
      fetch: async path => {
        assert.equal(path, `/runs/${run}/manifest.json`);
        return {ok: true, json: async () => data};
      },
    });
    await assert.rejects(main, error => error === rendered);
  });
}

test('a depth mask and a selected grid render on the actual Arcade 1 surface', async () => {
  const asset = new URL(
    `../benchmark/reviews/mobile-review-assets/${run}/train-arcade1-cool-high-right/surfaces.json`, import.meta.url,
  );
  const evidence = JSON.parse(readFileSync(asset, 'utf8'));
  const item = data.cases.find(row => row.id === evidence.id);
  const strokes = [];
  const context = {clearRect() {}, fill() {}, stroke(path) { strokes.push(path.path); }};
  const elements = new Map();
  const find = id => {
    if (!elements.has(id)) elements.set(id, {
      setAttribute() {}, replaceChildren(...options) { this.options = options; }, getContext: () => context,
    });
    return elements.get(id);
  };
  const factory = runInNewContext(definitions + '\nsurfaceReview', {
    document: {querySelector() {}},
    location: new URL('https://example.test/'),
    URLSearchParams,
    Option: class { constructor(text, value) { this.text = text; this.value = value; } },
    Path2D: class { constructor(path) { this.path = path; } rect() {} addPath() {} },
    fetch: async url => {
      assert.equal(url, item.surfaceEvidence);
      return {ok: true, json: async () => evidence};
    },
  });
  const viewer = factory(find, () => {});
  viewer.reset(item);
  await viewer.open('mask-1');
  assert.equal(viewer.selection().surface.id, 'mask-1');
  const active = viewer.selection().surface;
  const picker = find('surface-layer');
  picker.value = picker.options.find(option => option.text === 'Depth region 1').value;
  picker.onchange();
  assert.equal(viewer.selection().layer.part, 'mask-1~p1');
  assert.match(find('surface-receipt').textContent, /partial/);
  const selected = active.layers.find(layer => layer.status.includes('Candidate · selected'));
  picker.value = 2 + Object.keys(active.parts).length + active.layers.indexOf(selected);
  picker.onchange();
  assert.ok(strokes.includes(selected.path));
  assert.match(find('surface-receipt').textContent, /Candidate · selected/);
  find('next-surface').onclick();
  assert.notEqual(viewer.selection().surface.id, 'mask-1');
});
