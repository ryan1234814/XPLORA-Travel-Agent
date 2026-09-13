import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildPackList, categorizePackItem, splitPackItems } from './pack.ts';
import type { PackWeatherSource } from './pack.ts';

const ids = (items: ReturnType<typeof buildPackList>) => items.map((i) => i.label);
const UNIVERSAL = ['Refillable water bottle', 'Passport & travel documents'];

function weather(overrides: Partial<PackWeatherSource> = {}): PackWeatherSource {
  return {
    packing: [],
    temperature_c: { expected_low: 15, expected_high: 24 },
    conditions_summary: 'Pleasant and dry',
    activity_suggestions: [],
    ...overrides,
  };
}

describe('splitPackItems', () => {
  it('splits comma-separated text and trims', () => {
    assert.deepEqual(splitPackItems('Heavy jacket, Thermal layers'), ['Heavy jacket', 'Thermal layers']);
    assert.deepEqual(splitPackItems('Umbrella,  Sun hat , scarf'), ['Umbrella', 'Sun hat', 'scarf']);
  });

  it('keeps a single item untouched', () => {
    assert.deepEqual(splitPackItems('Sunscreen'), ['Sunscreen']);
  });

  it('drops empty segments', () => {
    assert.deepEqual(splitPackItems('Sunscreen,,Water'), ['Sunscreen', 'Water']);
    assert.deepEqual(splitPackItems(',,,,'), []);
  });
});

describe('categorizePackItem', () => {
  it('classifies by keyword into the four categories', () => {
    assert.equal(categorizePackItem('Compact umbrella'), 'Essentials');
    assert.equal(categorizePackItem('Refillable water bottle'), 'Essentials');
    assert.equal(categorizePackItem('Sunscreen'), 'Tech/Health');
    assert.equal(categorizePackItem('Lens cloth'), 'Tech/Health');
    assert.equal(categorizePackItem('Travel adapter'), 'Tech/Health');
    assert.equal(categorizePackItem('Passport & travel documents'), 'Docs');
    assert.equal(categorizePackItem('Visa & entry tickets'), 'Docs');
    assert.equal(categorizePackItem('Heavy jacket'), 'Clothing');
    assert.equal(categorizePackItem('Hiking boots'), 'Clothing');
  });

  it('does not misclassify waterproof gear as essentials or docs', () => {
    assert.equal(categorizePackItem('Waterproof shoes'), 'Clothing');
    assert.equal(categorizePackItem('Waterproof rain jacket'), 'Clothing');
  });
});

describe('buildPackList', () => {
  it('returns universal fallback items when weather is unavailable', () => {
    const items = buildPackList(null);
    assert.deepEqual(ids(items), UNIVERSAL);
    for (const item of items) {
      assert.ok(item.id, 'item id is set');
    }
  });

  it('always includes the universal items on top of LLM suggestions', () => {
    const items = buildPackList(weather({ packing: ['Comfortable walking shoes'] }));
    assert.deepEqual(ids(items), ['Comfortable walking shoes', ...UNIVERSAL]);
  });

  it('adds heavy layers when lows drop below 10°C', () => {
    const items = buildPackList(weather({ temperature_c: { expected_low: 2, expected_high: 8 } }));
    const labels = ids(items);
    assert.ok(labels.includes('Heavy jacket'));
    assert.ok(labels.includes('Thermal layers'));
    assert.ok(!labels.includes('Linen shirts'));
    assert.ok(!labels.includes('Sun hat'));
    assert.equal(items.find((i) => i.label === 'Heavy jacket')?.category, 'Clothing');
  });

  it('treats 10°C as the cold boundary (10 is not cold)', () => {
    const items = buildPackList(weather({ temperature_c: { expected_low: 10, expected_high: 16 } }));
    assert.ok(!ids(items).includes('Heavy jacket'));
  });

  it('adds linen and sun hat when highs exceed 28°C', () => {
    const items = buildPackList(weather({ temperature_c: { expected_low: 26, expected_high: 33 } }));
    const labels = ids(items);
    assert.ok(labels.includes('Linen shirts'));
    assert.ok(labels.includes('Sun hat'));
  });

  it('treats 28°C as the hot boundary (28 is not hot)', () => {
    const items = buildPackList(weather({ temperature_c: { expected_low: 22, expected_high: 28 } }));
    assert.ok(!ids(items).includes('Linen shirts'));
  });

  it('adds rain gear for rain, monsoon and humid conditions (case-insensitive)', () => {
    for (const summary of ['Light rain showers expected', 'Monsoon season begins', 'Very humid conditions', 'Rainy afternoons']) {
      const items = buildPackList(weather({ conditions_summary: summary }));
      const labels = ids(items);
      assert.ok(labels.includes('Compact umbrella'), summary);
      assert.ok(labels.includes('Waterproof shoes'), summary);
      assert.equal(items.find((i) => i.label === 'Compact umbrella')?.category, 'Essentials');
    }
  });

  it('skips rain gear on dry forecasts', () => {
    const items = buildPackList(weather({ conditions_summary: 'Sunny and dry, great for exploring' }));
    assert.ok(!ids(items).includes('Compact umbrella'));
    assert.ok(!ids(items).includes('Waterproof shoes'));
  });

  it('adds hiking boots when activities mention hiking or trekking', () => {
    const a = buildPackList(weather({ activity_suggestions: ['Hiking the alpine ridge at dawn', 'Evening food walk'] }));
    assert.ok(ids(a).includes('Hiking boots'));
    const b = buildPackList(weather({ activity_suggestions: ['TREKKING through the valley'] }));
    assert.ok(ids(b).includes('Hiking boots'));
    const c = buildPackList(weather({ activity_suggestions: ['Museum visits and city strolling'] }));
    assert.ok(!ids(c).includes('Hiking boots'));
  });

  it('adds a lens cloth for the Photography interest only', () => {
    const items = buildPackList(weather(), { interests: ['Photography', 'Gastronomy'] });
    assert.ok(ids(items).includes('Lens cloth'));
    const none = buildPackList(weather(), { interests: ['Gastronomy'] });
    assert.ok(!ids(none).includes('Lens cloth'));
  });

  it('adds compression socks only for the Intense pace', () => {
    assert.ok(ids(buildPackList(weather(), { pace: 'Intense' })).includes('Compression socks'));
    for (const pace of ['Relaxed', 'Moderate', 'Active', '']) {
      assert.ok(!ids(buildPackList(weather(), { pace })).includes('Compression socks'), pace);
    }
  });

  it('merges LLM packing, splits comma lists and dedupes case-insensitively', () => {
    const items = buildPackList(
      weather({
        packing: ['Umbrella, Sun hat', 'Sun Hat', 'light rain jacket', 'Sunscreen, lip balm'],
        conditions_summary: 'Humid monsoon',
        temperature_c: { expected_low: 5, expected_high: 20 },
      }),
      { pace: 'Intense', interests: ['Photography'] },
    );
    const labels = ids(items);
    assert.equal(labels.filter((l) => l.toLowerCase() === 'sun hat').length, 1, 'Sun hat deduped');
    assert.equal(labels.filter((l) => l.toLowerCase() === 'umbrella').length, 1, 'umbrella appears once');
    assert.ok(labels.includes('light rain jacket'));
    assert.ok(labels.includes('Sunscreen'));
    assert.ok(labels.includes('lip balm'));
    assert.ok(labels.includes('Heavy jacket')); // low 5 < 10 rule still applies
    assert.ok(labels.includes('Compression socks')); // Intense pace rule still applies
    assert.ok(labels.includes('Lens cloth')); // Photography rule still applies
  });

  it('dedupes rule items that the LLM already suggested', () => {
    const items = buildPackList(
      weather({ packing: ['Compact umbrella'], conditions_summary: 'Afternoon rain' }),
    );
    assert.equal(ids(items).filter((l) => l === 'Compact umbrella').length, 1);
  });

  it('leaves generic LLM copy in the Clothing bucket by default', () => {
    const items = buildPackList(weather({ packing: ['Standard travel wear based on temperature'] }));
    assert.equal(items.find((i) => i.label === 'Standard travel wear based on temperature')?.category, 'Clothing');
  });

  it('groups every produced item under one of the four categories', () => {
    const items = buildPackList(
      weather({
        packing: ['Passport', 'Sunscreen', 'Sandals', 'Umbrella, Phone charger'],
        conditions_summary: 'rain',
      }),
      { pace: 'Intense', interests: ['Photography'] },
    );
    const valid = new Set(['Essentials', 'Clothing', 'Tech/Health', 'Docs']);
    assert.ok(items.length > 0);
    for (const item of items) {
      assert.ok(valid.has(item.category), `${item.label} -> ${item.category}`);
    }
  });
});
