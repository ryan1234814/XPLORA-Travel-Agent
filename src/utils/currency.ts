// Client-side port of the currency + budget tables defined in the Python agents
// (agents/agents.py: _CURRENCY_MAP, _BUDGET_TIERS, _USD_TO_LOCAL_APPROX and the
// per-tier daily cost guide). Pure TS — no API calls, no backend dependency.

/** [symbol, code, name] — mirrors the Python (symbol, code, name) tuples. */
export type Currency = readonly [symbol: string, code: string, name: string];

export type BudgetTier = 'Essential' | 'Premier' | 'Elite' | 'Legendary';

export type CostCategory =
  | 'meal_breakfast'
  | 'meal_lunch'
  | 'meal_dinner'
  | 'attraction_entry'
  | 'local_transport'
  | 'taxi_ride'
  | 'coffee_snack';

// Destination keyword -> (symbol, code, name). Exact copy of _CURRENCY_MAP.
export const CURRENCY_MAP: Record<string, Currency> = {
  // Europe
  paris: ['€', 'EUR', 'Euro'],
  france: ['€', 'EUR', 'Euro'],
  london: ['£', 'GBP', 'British Pound'],
  uk: ['£', 'GBP', 'British Pound'],
  england: ['£', 'GBP', 'British Pound'],
  scotland: ['£', 'GBP', 'British Pound'],
  rome: ['€', 'EUR', 'Euro'],
  italy: ['€', 'EUR', 'Euro'],
  florence: ['€', 'EUR', 'Euro'],
  venice: ['€', 'EUR', 'Euro'],
  amsterdam: ['€', 'EUR', 'Euro'],
  netherlands: ['€', 'EUR', 'Euro'],
  berlin: ['€', 'EUR', 'Euro'],
  germany: ['€', 'EUR', 'Euro'],
  munich: ['€', 'EUR', 'Euro'],
  barcelona: ['€', 'EUR', 'Euro'],
  spain: ['€', 'EUR', 'Euro'],
  madrid: ['€', 'EUR', 'Euro'],
  lisbon: ['€', 'EUR', 'Euro'],
  portugal: ['€', 'EUR', 'Euro'],
  athens: ['€', 'EUR', 'Euro'],
  greece: ['€', 'EUR', 'Euro'],
  santorini: ['€', 'EUR', 'Euro'],
  zurich: ['CHF', 'CHF', 'Swiss Franc'],
  switzerland: ['CHF', 'CHF', 'Swiss Franc'],
  vienna: ['€', 'EUR', 'Euro'],
  austria: ['€', 'EUR', 'Euro'],
  prague: ['CZK', 'CZK', 'Czech Koruna'],
  czech: ['CZK', 'CZK', 'Czech Koruna'],
  budapest: ['HUF', 'HUF', 'Hungarian Forint'],
  hungary: ['HUF', 'HUF', 'Hungarian Forint'],
  copenhagen: ['kr', 'DKK', 'Danish Krone'],
  denmark: ['kr', 'DKK', 'Danish Krone'],
  stockholm: ['kr', 'SEK', 'Swedish Krona'],
  sweden: ['kr', 'SEK', 'Swedish Krona'],
  oslo: ['kr', 'NOK', 'Norwegian Krone'],
  norway: ['kr', 'NOK', 'Norwegian Krone'],
  dublin: ['€', 'EUR', 'Euro'],
  ireland: ['€', 'EUR', 'Euro'],
  moscow: ['₽', 'RUB', 'Russian Ruble'],
  russia: ['₽', 'RUB', 'Russian Ruble'],
  istanbul: ['₺', 'TRY', 'Turkish Lira'],
  turkey: ['₺', 'TRY', 'Turkish Lira'],
  croatia: ['€', 'EUR', 'Euro'],
  split: ['€', 'EUR', 'Euro'],
  // Asia
  tokyo: ['¥', 'JPY', 'Japanese Yen'],
  japan: ['¥', 'JPY', 'Japanese Yen'],
  kyoto: ['¥', 'JPY', 'Japanese Yen'],
  osaka: ['¥', 'JPY', 'Japanese Yen'],
  hokkaido: ['¥', 'JPY', 'Japanese Yen'],
  beijing: ['¥', 'CNY', 'Chinese Yuan'],
  china: ['¥', 'CNY', 'Chinese Yuan'],
  shanghai: ['¥', 'CNY', 'Chinese Yuan'],
  chengdu: ['¥', 'CNY', 'Chinese Yuan'],
  'hong kong': ['HK$', 'HKD', 'Hong Kong Dollar'],
  taiwan: ['NT$', 'TWD', 'Taiwan Dollar'],
  taipei: ['NT$', 'TWD', 'Taiwan Dollar'],
  bangkok: ['฿', 'THB', 'Thai Baht'],
  thailand: ['฿', 'THB', 'Thai Baht'],
  phuket: ['฿', 'THB', 'Thai Baht'],
  'chiang mai': ['฿', 'THB', 'Thai Baht'],
  bali: ['Rp', 'IDR', 'Indonesian Rupiah'],
  indonesia: ['Rp', 'IDR', 'Indonesian Rupiah'],
  singapore: ['S$', 'SGD', 'Singapore Dollar'],
  'kuala lumpur': ['RM', 'MYR', 'Malaysian Ringgit'],
  malaysia: ['RM', 'MYR', 'Malaysian Ringgit'],
  manila: ['₱', 'PHP', 'Philippine Peso'],
  philippines: ['₱', 'PHP', 'Philippine Peso'],
  seoul: ['₩', 'KRW', 'South Korean Won'],
  korea: ['₩', 'KRW', 'South Korean Won'],
  india: ['₹', 'INR', 'Indian Rupee'],
  delhi: ['₹', 'INR', 'Indian Rupee'],
  mumbai: ['₹', 'INR', 'Indian Rupee'],
  jaipur: ['₹', 'INR', 'Indian Rupee'],
  goa: ['₹', 'INR', 'Indian Rupee'],
  munnar: ['₹', 'INR', 'Indian Rupee'],
  kerala: ['₹', 'INR', 'Indian Rupee'],
  kochi: ['₹', 'INR', 'Indian Rupee'],
  cochin: ['₹', 'INR', 'Indian Rupee'],
  varanasi: ['₹', 'INR', 'Indian Rupee'],
  agra: ['₹', 'INR', 'Indian Rupee'],
  udaipur: ['₹', 'INR', 'Indian Rupee'],
  rishikesh: ['₹', 'INR', 'Indian Rupee'],
  ladakh: ['₹', 'INR', 'Indian Rupee'],
  kashmir: ['₹', 'INR', 'Indian Rupee'],
  himachal: ['₹', 'INR', 'Indian Rupee'],
  manali: ['₹', 'INR', 'Indian Rupee'],
  shimla: ['₹', 'INR', 'Indian Rupee'],
  darjeeling: ['₹', 'INR', 'Indian Rupee'],
  ooty: ['₹', 'INR', 'Indian Rupee'],
  coorg: ['₹', 'INR', 'Indian Rupee'],
  pondicherry: ['₹', 'INR', 'Indian Rupee'],
  hampi: ['₹', 'INR', 'Indian Rupee'],
  andaman: ['₹', 'INR', 'Indian Rupee'],
  bangalore: ['₹', 'INR', 'Indian Rupee'],
  bengaluru: ['₹', 'INR', 'Indian Rupee'],
  chennai: ['₹', 'INR', 'Indian Rupee'],
  hyderabad: ['₹', 'INR', 'Indian Rupee'],
  kolkata: ['₹', 'INR', 'Indian Rupee'],
  hanoi: ['₫', 'VND', 'Vietnamese Dong'],
  vietnam: ['₫', 'VND', 'Vietnamese Dong'],
  'ho chi minh': ['₫', 'VND', 'Vietnamese Dong'],
  cambodia: ['៛', 'KHR', 'Cambodian Riel'],
  'phnom penh': ['៛', 'KHR', 'Cambodian Riel'],
  nepal: ['NPR', 'NPR', 'Nepalese Rupee'],
  'sri lanka': ['Rs', 'LKR', 'Sri Lankan Rupee'],
  // Middle East
  dubai: ['د.إ', 'AED', 'UAE Dirham'],
  uae: ['د.إ', 'AED', 'UAE Dirham'],
  'abu dhabi': ['د.إ', 'AED', 'UAE Dirham'],
  qatar: ['QR', 'QAR', 'Qatari Riyal'],
  doha: ['QR', 'QAR', 'Qatari Riyal'],
  'saudi arabia': ['﷼', 'SAR', 'Saudi Riyal'],
  riyadh: ['﷼', 'SAR', 'Saudi Riyal'],
  oman: ['ر.ع', 'OMR', 'Omani Rial'],
  bahrain: ['BD', 'BHD', 'Bahraini Dinar'],
  israel: ['₪', 'ILS', 'Israeli Shekel'],
  'tel aviv': ['₪', 'ILS', 'Israeli Shekel'],
  jordan: ['JD', 'JOD', 'Jordanian Dinar'],
  // Oceania
  sydney: ['A$', 'AUD', 'Australian Dollar'],
  australia: ['A$', 'AUD', 'Australian Dollar'],
  melbourne: ['A$', 'AUD', 'Australian Dollar'],
  'gold coast': ['A$', 'AUD', 'Australian Dollar'],
  'new zealand': ['NZ$', 'NZD', 'New Zealand Dollar'],
  auckland: ['NZ$', 'NZD', 'New Zealand Dollar'],
  queenstown: ['NZ$', 'NZD', 'New Zealand Dollar'],
  // Americas
  'new york': ['$', 'USD', 'US Dollar'],
  'los angeles': ['$', 'USD', 'US Dollar'],
  'san francisco': ['$', 'USD', 'US Dollar'],
  miami: ['$', 'USD', 'US Dollar'],
  'las vegas': ['$', 'USD', 'US Dollar'],
  chicago: ['$', 'USD', 'US Dollar'],
  canada: ['C$', 'CAD', 'Canadian Dollar'],
  toronto: ['C$', 'CAD', 'Canadian Dollar'],
  vancouver: ['C$', 'CAD', 'Canadian Dollar'],
  mexico: ['MX$', 'MXN', 'Mexican Peso'],
  cancun: ['MX$', 'MXN', 'Mexican Peso'],
  'playa del carmen': ['MX$', 'MXN', 'Mexican Peso'],
  brazil: ['R$', 'BRL', 'Brazilian Real'],
  'rio de janeiro': ['R$', 'BRL', 'Brazilian Real'],
  'sao paulo': ['R$', 'BRL', 'Brazilian Real'],
  argentina: ['$', 'ARS', 'Argentine Peso'],
  'buenos aires': ['$', 'ARS', 'Argentine Peso'],
  peru: ['S/', 'PEN', 'Peruvian Sol'],
  lima: ['S/', 'PEN', 'Peruvian Sol'],
  cuzco: ['S/', 'PEN', 'Peruvian Sol'],
  colombia: ['COL$', 'COP', 'Colombian Peso'],
  bogota: ['COL$', 'COP', 'Colombian Peso'],
  cartagena: ['COL$', 'COP', 'Colombian Peso'],
  chile: ['CL$', 'CLP', 'Chilean Peso'],
  santiago: ['CL$', 'CLP', 'Chilean Peso'],
  'costa rica': ['₡', 'CRC', 'Costa Rican Colón'],
  caribbean: ['$', 'USD', 'US Dollar'],
  cuba: ['CUC', 'CUC', 'Cuban Peso'],
  // Africa
  'cape town': ['R', 'ZAR', 'South African Rand'],
  'south africa': ['R', 'ZAR', 'South African Rand'],
  johannesburg: ['R', 'ZAR', 'South African Rand'],
  cairo: ['E£', 'EGP', 'Egyptian Pound'],
  egypt: ['E£', 'EGP', 'Egyptian Pound'],
  marrakech: ['د.م.', 'MAD', 'Moroccan Dirham'],
  morocco: ['د.م.', 'MAD', 'Moroccan Dirham'],
  kenya: ['KSh', 'KES', 'Kenyan Shilling'],
  nairobi: ['KSh', 'KES', 'Kenyan Shilling'],
  tanzania: ['TSh', 'TZS', 'Tanzanian Shilling'],
  zanzibar: ['TSh', 'TZS', 'Tanzanian Shilling'],
  ethiopia: ['Br', 'ETB', 'Ethiopian Birr'],
  accra: ['GH₵', 'GHS', 'Ghanaian Cedi'],
  ghana: ['GH₵', 'GHS', 'Ghanaian Cedi'],
  // Default
  usa: ['$', 'USD', 'US Dollar'],
  'united states': ['$', 'USD', 'US Dollar'],
};

// Budget tier -> approximate USD range per day per person (fallback reference).
// Exact copy of _BUDGET_TIERS.
export const BUDGET_TIERS: Record<BudgetTier, readonly [number, number]> = {
  Essential: [50, 100],
  Premier: [150, 300],
  Elite: [300, 600],
  Legendary: [600, 1500],
};

// Approximate USD -> local currency multipliers (rough, for fallback pricing).
// Exact copy of _USD_TO_LOCAL_APPROX.
export const USD_TO_LOCAL_APPROX: Record<string, number> = {
  EUR: 0.92, GBP: 0.79, CHF: 0.88, JPY: 150.0, CNY: 7.25,
  HKD: 7.82, TWD: 32.0, THB: 35.0, IDR: 15800.0, SGD: 1.35,
  MYR: 4.70, PHP: 56.0, KRW: 1350.0, INR: 83.5, VND: 24500.0,
  KHR: 4100.0, LKR: 310.0, NPR: 133.5,
  AED: 3.67, QAR: 3.64, SAR: 3.75, OMR: 0.385, BHD: 0.376,
  ILS: 3.65, JOD: 0.709,
  AUD: 1.55, NZD: 1.68,
  CAD: 1.36, MXN: 17.0, BRL: 5.0, ARS: 350.0,
  PEN: 3.70, COP: 3950.0, CLP: 920.0, CRC: 520.0, CUC: 1.0,
  ZAR: 18.5, EGP: 48.0, MAD: 10.0, KES: 153.0, TZS: 2500.0,
  ETB: 56.0, GHS: 15.5, DKK: 6.85, SEK: 10.5, NOK: 10.8,
  CZK: 23.0, HUF: 360.0, TRY: 32.5, RUB: 92.0, RSD: 108.0,
};

// Per-person daily cost ranges (USD) for each budget tier, by category.
// Exact copy of the `tiers` dict in _get_budget_cost_guide.
export const COST_TIERS: Record<BudgetTier, Record<CostCategory, readonly [number, number]>> = {
  Essential: {
    meal_breakfast: [3, 8], meal_lunch: [5, 12], meal_dinner: [8, 18],
    attraction_entry: [3, 15], local_transport: [1, 5],
    taxi_ride: [3, 10], coffee_snack: [1, 3],
  },
  Premier: {
    meal_breakfast: [8, 20], meal_lunch: [15, 35], meal_dinner: [25, 60],
    attraction_entry: [10, 40], local_transport: [3, 10],
    taxi_ride: [8, 25], coffee_snack: [3, 8],
  },
  Elite: {
    meal_breakfast: [20, 50], meal_lunch: [35, 80], meal_dinner: [60, 150],
    attraction_entry: [25, 80], local_transport: [8, 20],
    taxi_ride: [15, 50], coffee_snack: [5, 15],
  },
  Legendary: {
    meal_breakfast: [50, 120], meal_lunch: [80, 200], meal_dinner: [150, 400],
    attraction_entry: [50, 200], local_transport: [15, 40],
    taxi_ride: [30, 100], coffee_snack: [10, 30],
  },
};

/** Display labels for each cost category (used in the FX tracker table). */
export const COST_CATEGORY_LABELS: Record<CostCategory, string> = {
  meal_breakfast: 'Breakfast',
  meal_lunch: 'Lunch',
  meal_dinner: 'Dinner',
  attraction_entry: 'Attraction',
  local_transport: 'Transport',
  taxi_ride: 'Taxi',
  coffee_snack: 'Coffee',
};

/** True when the string is one of the four supported budget tiers. */
export function isBudgetTier(value: string): value is BudgetTier {
  return value === 'Essential' || value === 'Premier' || value === 'Elite' || value === 'Legendary';
}

/**
 * Return (symbol, code, name) for the destination's local currency.
 * Tries the longest keyword match first; falls back to USD.
 */
export function getCurrencyForDestination(destination: string): Currency {
  const destLower = (destination || '').toLowerCase();
  for (const key of Object.keys(CURRENCY_MAP).sort((a, b) => b.length - a.length)) {
    if (destLower.includes(key)) return CURRENCY_MAP[key];
  }
  return ['$', 'USD', 'US Dollar'];
}

/** Rough USD -> local multiplier for a currency code (1.0 when unknown). */
export function getExchangeMultiplier(code: string): number {
  return USD_TO_LOCAL_APPROX[code] ?? 1;
}

/** Format a rounded integer with thousands separators, mirroring Python's f"{n:,}". */
export function formatLocalNumber(n: number): string {
  return Math.round(n).toLocaleString('en-US');
}

/**
 * Build a price_range string in the destination's local currency.
 * Same math as agents/agents.py `_format_price_range`.
 */
export function formatPriceRange(destination: string, budgetTier: string, duration: number): string {
  const [symbol, code, name] = getCurrencyForDestination(destination);
  const tierBounds = isBudgetTier(budgetTier) ? BUDGET_TIERS[budgetTier] : BUDGET_TIERS.Premier;
  const multiplier = getExchangeMultiplier(code);

  const lowUsd = tierBounds[0] * duration;
  const highUsd = tierBounds[1] * duration;

  const lowLocal = Math.round(lowUsd * multiplier);
  const highLocal = Math.round(highUsd * multiplier);

  return `${symbol}${formatLocalNumber(lowLocal)} - ${symbol}${formatLocalNumber(highLocal)} ${name}`;
}
