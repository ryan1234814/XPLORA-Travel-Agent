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
      <div className="main-gradient min-h-screen flex items-center justify-center text-slate-400">
        <Loader2 className="w-5 h-5 animate-spin mr-2" />
        <span className="text-[10px] font-bold uppercase tracking-[0.3em]">Checking your session</span>
      </div>
    );
  }
  if (user === null) return <Navigate to="/login" replace />;
  return <>{children}</>;
}
