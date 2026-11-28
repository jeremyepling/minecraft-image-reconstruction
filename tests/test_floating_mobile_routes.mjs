import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {runInNewContext} from 'node:vm';

const site = new URL('../review-site/', import.meta.url);
const run = 'floating-v1-guarded-refit';
const markup = readFileSync(new URL('index.html', site), 'utf8');
const script = readFileSync(new URL('app.js', site), 'utf8');
const data = JSON.parse(readFileSync(new URL(`runs/${run}/manifest.json`, site), 'utf8'));

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
    const main = runInNewContext(script.slice(0, script.indexOf('\nmain().catch')) + '\nmain()', {
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
