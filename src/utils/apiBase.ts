import axios from 'axios';

// Central resolution of the backend API base URL.
//
// In production the backend is reached through a Vercel rewrite (see
// vercel.json) so the HTTP-only session cookie stays FIRST-PARTY. If we called
// the Render backend directly (onrender.com) from the Vercel frontend
// (vercel.app) the cookie would be a third-party cookie and browsers would
// drop it — every signed-in request would then look anonymous. Proxying via
// Vercel fixes that, so in production we use relative URLs.
//
// VITE_API_BASE_URL is honoured only in local development, where there is no
// Vercel proxy and the backend runs on localhost.
const envUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;

export const API_BASE_URL = import.meta.env.PROD
  ? ''
  : (envUrl || 'http://localhost:8000');

// Fire-and-forget warm-up. Render's free tier sleeps after ~15 min idle and the
// next request cold-starts in ~50-60s. Pinging an API endpoint on app load
// overlaps that cold start with the user reading the page, instead of blocking
// their first real action. Errors are ignored — this is best-effort only.
export function warmBackend(): void {
  const healthUrl = import.meta.env.PROD
    ? '/api/auth/me'          // proxied through Vercel; 401 here is fine, it still wakes Render
    : (API_BASE_URL ? `${API_BASE_URL}/` : '');
  if (!healthUrl) return;
  axios.get(healthUrl, { timeout: 120000 }).catch(() => {});
}
