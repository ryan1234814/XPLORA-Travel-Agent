import type { ItineraryData, MobilityData, WeatherData } from '../App';

/** A stored plan is the raw agent_outputs map: { itinerary_planner: {...}, ... } */
export interface StoredPlanPayload {
  itinerary_planner?: unknown;
  transport_mobility?: unknown;
  weather_analyst?: unknown;
  local_expert?: unknown;
  [key: string]: unknown;
}

export interface ParsedPlan {
  itinerary: ItineraryData | null;
  mobility: MobilityData | null;
  weather: WeatherData | null;
  localExpert: unknown;
}

/**
 * Agent sections may arrive as objects, as `{ output: {...} }` wrappers, or as
 * strings with JSON embedded in prose. Shared by the live generator and the
 * dashboard's "reopen a saved trip" path so both read a plan identically.
 */
export function parseAgentField<T = unknown>(field: unknown): T | null {
  if (field == null) return null;
  const record = field as Record<string, unknown>;
  const obj = typeof field === 'object' && 'output' in record ? record.output : field;
  if (typeof obj !== 'string') return obj as T;
  const match = obj.match(/\{[\s\S]*\}/);
  if (!match) return obj as unknown as T;
  try {
    return JSON.parse(match[0]) as T;
  } catch {
    return obj as unknown as T;
  }
}

export function parseStoredPlan(payload: StoredPlanPayload | null | undefined): ParsedPlan {
  const itinerary = parseAgentField(payload?.itinerary_planner) as ItineraryData | null;
  return {
    itinerary: itinerary && typeof itinerary === 'object' ? itinerary : null,
    mobility: parseAgentField(payload?.transport_mobility) as MobilityData | null,
    weather: parseAgentField(payload?.weather_analyst) as WeatherData | null,
    localExpert: parseAgentField(payload?.local_expert),
  };
}

/** Human label for the day count shown on dashboard cards. */
export function describeDuration(days?: number | null): string {
  if (!days || days < 1) return '';
  return `${days} ${days === 1 ? 'day' : 'days'}`;
}

export function formatSavedOn(iso?: string | null): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}
