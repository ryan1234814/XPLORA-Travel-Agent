// Central resolution of the backend API base URL.
//
// In production this value is baked in at BUILD time from VITE_API_BASE_URL —
// set it to the Render backend URL (e.g. https://xplora-backend.onrender.com)
// in the Vercel project settings, and redeploy whenever the backend URL changes.
const envUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;

if (import.meta.env.PROD && !envUrl) {
  // Fail loudly: without this, /api/* calls would silently hit the frontend
  // origin (Vercel) and return index.html instead of JSON.
  console.error(
    '[XPLORA] VITE_API_BASE_URL is not set in this production build. ' +
    'API calls will fail. Set it to the backend URL (e.g. https://xplora-backend.onrender.com) ' +
    'in Vercel environment variables and redeploy.'
  );
}

export const API_BASE_URL =
  envUrl || (import.meta.env.PROD ? '' : 'http://localhost:8000');
