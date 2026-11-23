import type { ReactNode } from 'react';
import { Loader2 } from 'lucide-react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../utils/auth';

/**
 * Guards the routes that need an account. `user === undefined` means the
 * cookie session has not been checked yet, so we wait rather than flash a
 * redirect at a user who is in fact signed in.
 */
export default function AuthGate({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  if (user === undefined) {
    return (
      <div className="flex min-h-screen items-center justify-center gap-2.5 bg-black text-slate-500">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span className="text-[11px]">Checking your session…</span>
      </div>
    );
  }
  if (user === null) return <Navigate to="/login" replace />;
  return <>{children}</>;
}
