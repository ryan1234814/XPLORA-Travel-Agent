import axios, { AxiosError } from 'axios';
import { useSyncExternalStore } from 'react';
import { API_BASE_URL } from '../utils/apiBase';

export interface AuthUser {
  id: number;
  email: string;
  display_name?: string | null;
  created_at?: string;
}

export interface AuthState {
  /** undefined = not fetched yet, null = fetched and signed out */
  user: AuthUser | null | undefined;
}

export interface SavedItinerarySummary {
  id: number;
  origin?: string | null;
  destination?: string | null;
  duration?: number | null;
  budget?: string | null;
  interests?: string[] | null;
  travel_dates?: string | null;
  group_size?: number | null;
  group_type?: string | null;
  dietary_requirements?: string[] | null;
  accessibility?: string[] | null;
  pace?: string | null;
  accommodation_preference?: string | null;
  occasion?: string | null;
  title?: string | null;
  created_at?: string;
}

export interface SavedItinerary extends SavedItinerarySummary {
  itinerary_data: Record<string, unknown>;
  user_id?: number | null;
}

const client = axios.create({
  baseURL: API_BASE_URL,
  // The session lives in an HTTP-only cookie, so every call must carry it.
  withCredentials: true,
  timeout: 30000,
});

let state: AuthState = { user: undefined };
const listeners = new Set<() => void>();

function set(next: Partial<AuthState>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useAuth(): AuthState {
  return useSyncExternalStore(subscribe, () => state);
}

export function authUser(): AuthUser | null | undefined {
  return state.user;
}

/** Turns an Axios error into something the UI can show. */
export function apiError(err: unknown): string {
  const ax = err as AxiosError<{ detail?: unknown }>;
  const detail = ax?.response?.data?.detail;
  if (typeof detail === 'string' && detail) return detail;
  if (Array.isArray(detail)) {
    const first = detail[0] as { msg?: string };
    if (first?.msg) return first.msg;
  }
  if (ax?.code === 'ECONNABORTED') return 'The server took too long to respond.';
  if (!ax?.response) return 'Cannot reach the server. Is the backend running?';
  return `Request failed (${ax.response.status})`;
}

/** Loads the signed-in user from the cookie session. Always resolves. */
export async function refreshAuth(): Promise<AuthUser | null> {
  try {
    const { data } = await client.get('/api/auth/me');
    const user = (data?.user ?? null) as AuthUser | null;
    set({ user });
    return user;
  } catch {
    set({ user: null });
    return null;
  }
}

export async function signUp(input: {
  email: string;
  password: string;
  confirmPassword: string;
  displayName?: string;
}): Promise<AuthUser> {
  const { data } = await client.post('/api/auth/signup', {
    email: input.email,
    password: input.password,
    confirm_password: input.confirmPassword,
    display_name: input.displayName || null,
  });
  const user = data.user as AuthUser;
  set({ user });
  return user;
}

export async function signIn(email: string, password: string): Promise<AuthUser> {
  const { data } = await client.post('/api/auth/login', { email, password });
  const user = data.user as AuthUser;
  set({ user });
  return user;
}

export async function signOut(): Promise<void> {
  try {
    await client.post('/api/auth/logout');
  } finally {
    set({ user: null });
  }
}

export async function fetchSavedItineraries(limit = 50): Promise<SavedItinerarySummary[]> {
  const { data } = await client.get('/api/auth/itineraries', { params: { limit } });
  return (data?.itineraries ?? []) as SavedItinerarySummary[];
}

export async function fetchSavedItinerary(id: number): Promise<SavedItinerary> {
  const { data } = await client.get(`/api/auth/itineraries/${id}`);
  return data as SavedItinerary;
}

export async function deleteSavedItinerary(id: number): Promise<void> {
  await client.delete(`/api/auth/itineraries/${id}`);
}

/**
 * The itinerary generator runs for minutes and must not send an AbortSignal we
 * can't control, so it uses its own long-timeout call rather than `client`.
 */
export async function postWithSession<T = unknown>(path: string, body: unknown, timeoutMs: number): Promise<T> {
  const { data } = await client.post<T>(path, body, { timeout: timeoutMs });
  return data;
}
