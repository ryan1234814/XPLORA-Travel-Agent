// Rule-based packing intelligence for the Smart Pack tracker.
// Pure TS (no React / DOM) so it can be unit-tested in isolation.
//
// Builds a deduplicated, categorized checklist by merging the weather analyst's
// LLM packing suggestions with deterministic weather / interest / pace rules.

export type PackCategory = 'Essentials' | 'Clothing' | 'Tech/Health' | 'Docs';

export interface PackItem {
  id: string;
  label: string;
  category: PackCategory;
}

/** Minimal structural view of WeatherData that the pack rules depend on. */
export interface PackWeatherSource {
  packing?: string[];
  temperature_c?: {
    expected_low?: number;
    expected_high?: number;
  };
  conditions_summary?: string;
  activity_suggestions?: string[];
}

export interface PackOptions {
  pace?: string;
  interests?: string[];
}

/** Items that are sensible to pack no matter what the forecast says. */
const UNIVERSAL_ITEMS: ReadonlyArray<string> = ['Refillable water bottle', 'Passport & travel documents'];

/** Split items that arrive as comma-separated text (e.g. "Heavy jacket, Thermal layers"). */
export function splitPackItems(raw: string): string[] {
  return raw
    .split(/,/g)
    .map((s) => s.trim())
    .filter(Boolean);
}

const DOCS_RE = /passport|visa|ticket|insurance|document|itinerary|booking|reservation|\bid\b|identity|driver|vaccinat|cash|credit|debit|embassy/i;
const TECH_HEALTH_RE = /sunscreen|adapt|charger|power\s?bank|battery|camera|lens|earplug|headphone|medication|medicine|first[- ]?aid|plaster|sanitizer|repellent|insect|sim\s?card|electronics|gadget|phone|flashlight|torch|skincare|moisturizer|lip\s?balm|memory\s?card|sd\s?card|cable|mirror/i;
const ESSENTIALS_RE = /umbrella|poncho|rain\s?cover|bottle|water(?!proof)|snack|insulated|day\s?pack|packing\s?cube|toiletries|wipes/i;

/** Classify a single packed item into one of the four checklist categories. */
export function categorizePackItem(item: string): PackCategory {
  const t = item.toLowerCase();
  if (DOCS_RE.test(t)) return 'Docs';
  if (TECH_HEALTH_RE.test(t)) return 'Tech/Health';
  if (ESSENTIALS_RE.test(t)) return 'Essentials';
  return 'Clothing';
}

/**
 * Build the deduplicated Smart Pack checklist for a destination's weather.
 *
 * - Always includes the universal fallback items (works with `weather = null`).
 * - Merges `weather.packing` (LLM output) with deterministic rules:
 *   cold lows add heavy layers, hot highs add linen/sun hat, rain/monsoon/humid
 *   conditions add an umbrella + waterproof shoes, hiking suggestions add
 *   boots, the Photography interest adds a lens cloth and the Intense pace
 *   adds compression socks.
 */
export function buildPackList(weather: PackWeatherSource | null | undefined, options: PackOptions = {}): PackItem[] {
  const pace = options.pace || '';
  const interests = options.interests || [];

  const low = weather?.temperature_c?.expected_low;
  const high = weather?.temperature_c?.expected_high;
  const conditions = (weather?.conditions_summary || '').toLowerCase();
  const activityText = (weather?.activity_suggestions || []).join(' ').toLowerCase();

  const extras: string[] = [...UNIVERSAL_ITEMS];
  if (low != null && low < 10) {
    extras.push('Heavy jacket', 'Thermal layers');
  }
  if (high != null && high > 28) {
    extras.push('Linen shirts', 'Sun hat');
  }
  if (conditions && /rain|monsoon|humid/.test(conditions)) {
    extras.push('Compact umbrella', 'Waterproof shoes');
  }
  if (activityText && /hiking|trek/.test(activityText)) {
    extras.push('Hiking boots');
  }
  if (interests.includes('Photography')) {
    extras.push('Lens cloth');
  }
  if (pace === 'Intense') {
    extras.push('Compression socks');
  }

  const seen = new Set<string>();
  const merged: PackItem[] = [];
  const push = (raw: string) => {
    for (const piece of splitPackItems(raw)) {
      const id = piece.toLowerCase();
      if (!piece || seen.has(id)) continue;
      seen.add(id);
      merged.push({ id, label: piece, category: categorizePackItem(piece) });
    }
  };

  (weather?.packing || []).forEach(push);
  extras.forEach(push);
  return merged;
}
