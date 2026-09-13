import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  BUDGET_TIERS,
  COST_TIERS,
  CURRENCY_MAP,
  USD_TO_LOCAL_APPROX,
  formatLocalNumber,
  formatPriceRange,
  getCurrencyForDestination,
  getExchangeMultiplier,
  isBudgetTier,
} from './currency.ts';

describe('getCurrencyForDestination', () => {
  it('maps well-known cities to their local currency', () => {
    assert.deepEqual(getCurrencyForDestination('Paris'), ['€', 'EUR', 'Euro']);
    assert.deepEqual(getCurrencyForDestination('London'), ['£', 'GBP', 'British Pound']);
    assert.deepEqual(getCurrencyForDestination('Kyoto'), ['¥', 'JPY', 'Japanese Yen']);
    assert.deepEqual(getCurrencyForDestination('Delhi'), ['₹', 'INR', 'Indian Rupee']);
    assert.deepEqual(getCurrencyForDestination('Bangkok'), ['฿', 'THB', 'Thai Baht']);
    assert.deepEqual(getCurrencyForDestination('Sydney'), ['A$', 'AUD', 'Australian Dollar']);
    assert.deepEqual(getCurrencyForDestination('Toronto'), ['C$', 'CAD', 'Canadian Dollar']);
    assert.deepEqual(getCurrencyForDestination('Sao Paulo'), ['R$', 'BRL', 'Brazilian Real']);
    assert.deepEqual(getCurrencyForDestination('Marrakech'), ['د.م.', 'MAD', 'Moroccan Dirham']);
    assert.deepEqual(getCurrencyForDestination('Cape Town'), ['R', 'ZAR', 'South African Rand']);
  });

  it('falls back to country/region keywords', () => {
    assert.deepEqual(getCurrencyForDestination('Indonesia'), ['Rp', 'IDR', 'Indonesian Rupiah']);
    assert.deepEqual(getCurrencyForDestination('Egypt'), ['E£', 'EGP', 'Egyptian Pound']);
    assert.deepEqual(getCurrencyForDestination('United States'), ['$', 'USD', 'US Dollar']);
    assert.deepEqual(getCurrencyForDestination('USA'), ['$', 'USD', 'US Dollar']);
    assert.deepEqual(getCurrencyForDestination('Spain'), ['€', 'EUR', 'Euro']);
  });

  it('is case-insensitive', () => {
    assert.deepEqual(getCurrencyForDestination('PARIS'), ['€', 'EUR', 'Euro']);
    assert.deepEqual(getCurrencyForDestination('dubai'), ['د.إ', 'AED', 'UAE Dirham']);
  });

  it('prefers the longest matching keyword over an earlier short one', () => {
    // 'uk' is a map key, but 'new york' is longer — the USD match must win.
    assert.deepEqual(getCurrencyForDestination('New York, UK'), ['$', 'USD', 'US Dollar']);
    // Both 'london' and 'uk' resolve to GBP anyway.
    assert.deepEqual(getCurrencyForDestination('London, UK'), ['£', 'GBP', 'British Pound']);
  });

  it('handles multi-word destinations', () => {
    assert.deepEqual(getCurrencyForDestination('Kuala Lumpur'), ['RM', 'MYR', 'Malaysian Ringgit']);
    assert.deepEqual(getCurrencyForDestination('Ho Chi Minh City'), ['₫', 'VND', 'Vietnamese Dong']);
    assert.deepEqual(getCurrencyForDestination('Playa del Carmen'), ['MX$', 'MXN', 'Mexican Peso']);
    assert.deepEqual(getCurrencyForDestination('Chiang Mai'), ['฿', 'THB', 'Thai Baht']);
  });

  it('falls back to USD for unknown destinations', () => {
    assert.deepEqual(getCurrencyForDestination('Atlantis'), ['$', 'USD', 'US Dollar']);
    assert.deepEqual(getCurrencyForDestination(''), ['$', 'USD', 'US Dollar']);
  });

  it('keeps India hub cities on the rupee', () => {
    for (const city of ['Mumbai', 'Jaipur', 'Goa', 'Kerala', 'Kochi', 'Agra', 'Ladakh', 'Manali', 'Ooty', 'Hampi', 'Bangalore', 'Hyderabad']) {
      assert.deepEqual(getCurrencyForDestination(city), ['₹', 'INR', 'Indian Rupee'], city);
    }
  });
});

describe('getExchangeMultiplier', () => {
  it('returns the stored approximate multipliers', () => {
    assert.equal(getExchangeMultiplier('JPY'), 150.0);
    assert.equal(getExchangeMultiplier('INR'), 83.5);
    assert.equal(getExchangeMultiplier('OMR'), 0.385);
    assert.equal(getExchangeMultiplier('EUR'), 0.92);
    assert.equal(getExchangeMultiplier('IDR'), 15800.0);
  });

  it('defaults to 1 for unknown codes', () => {
    assert.equal(getExchangeMultiplier('XXX'), 1);
    assert.equal(getExchangeMultiplier(''), 1);
  });

  it('covers every currency code used by CURRENCY_MAP (USD defaults to 1)', () => {
    const codes = new Set(Object.values(CURRENCY_MAP).map(([, code]) => code));
    for (const code of codes) {
      if (code in USD_TO_LOCAL_APPROX) continue;
      // Mirror of Python's `_USD_TO_LOCAL_APPROX.get(code, 1.0)`: USD is not
      // listed on purpose and resolves to the default 1.0 multiplier.
      assert.equal(code, 'USD', `only USD may lack a stored multiplier (got ${code})`);
      assert.equal(getExchangeMultiplier(code), 1);
    }
  });
});

describe('formatLocalNumber', () => {
  it('formats with thousands separators like Python f"{n:,}"', () => {
    assert.equal(formatLocalNumber(0), '0');
    assert.equal(formatLocalNumber(88), '88');
    assert.equal(formatLocalNumber(37575), '37,575');
    assert.equal(formatLocalNumber(135000), '135,000');
    assert.equal(formatLocalNumber(1000000), '1,000,000');
    assert.equal(formatLocalNumber(5.6), '6');
  });
});

describe('formatPriceRange', () => {
  it('matches the Python math for Delhi / Premier / 3 days', () => {
    assert.equal(formatPriceRange('Delhi', 'Premier', 3), '₹37,575 - ₹75,150 Indian Rupee');
  });

  it('matches the Python math for Kyoto / Premier / 3 days', () => {
    assert.equal(formatPriceRange('Kyoto', 'Premier', 3), '¥67,500 - ¥135,000 Japanese Yen');
  });

  it('scales with duration', () => {
    assert.equal(formatPriceRange('Paris', 'Essential', 7), '€322 - €644 Euro');
    assert.equal(formatPriceRange('London', 'Legendary', 1), '£474 - £1,185 British Pound');
    assert.equal(formatPriceRange('Berlin', 'Premier', 10), '€1,380 - €2,760 Euro');
  });

  it('rounds fractional local totals the same way Python does', () => {
    // 450 * 3.67 = 1651.5 -> round() = 1652; 900 * 3.67 = 3303
    assert.equal(formatPriceRange('Dubai', 'Premier', 3), 'د.إ1,652 - د.إ3,303 UAE Dirham');
  });

  it('falls back to Premier tier and USD for unknown inputs', () => {
    assert.equal(formatPriceRange('Nowhere', 'NotATier', 3), '$450 - $900 US Dollar');
  });
});

describe('tables', () => {
  it('exposes the four budget tiers with their daily USD bounds', () => {
    assert.deepEqual(BUDGET_TIERS.Essential, [50, 100]);
    assert.deepEqual(BUDGET_TIERS.Premier, [150, 300]);
    assert.deepEqual(BUDGET_TIERS.Elite, [300, 600]);
    assert.deepEqual(BUDGET_TIERS.Legendary, [600, 1500]);
  });

  it('exposes per-category daily cost ranges for every tier', () => {
    assert.deepEqual(COST_TIERS.Premier.meal_dinner, [25, 60]);
    assert.deepEqual(COST_TIERS.Essential.meal_breakfast, [3, 8]);
    assert.deepEqual(COST_TIERS.Legendary.taxi_ride, [30, 100]);
    assert.deepEqual(COST_TIERS.Elite.coffee_snack, [5, 15]);
  });

  it('recognizes only the four supported tier names', () => {
    for (const tier of ['Essential', 'Premier', 'Elite', 'Legendary']) {
      assert.ok(isBudgetTier(tier), tier);
    }
    assert.ok(!isBudgetTier('budget'));
    assert.ok(!isBudgetTier(''));
    assert.ok(!isBudgetTier('Premium'));
  });
});
