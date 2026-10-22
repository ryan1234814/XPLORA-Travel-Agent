import { useState, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Loader2, LogIn, Mail, Lock } from 'lucide-react';
import AuthShell, { Field, FormError, inputClass } from '../components/AuthShell';
import { apiError, signIn } from '../utils/auth';

export default function Login() {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await signIn(email.trim(), password);
      // Signing in lands on the dashboard — that is where the saved trips live.
      navigate('/dashboard', { replace: true });
    } catch (err) {
      setError(apiError(err));
      setBusy(false);
    }
  };

  return (
    <AuthShell
      eyebrow="Welcome back"
      title="Sign in to XPLORA"
      subtitle="Your account keeps every itinerary you have generated, so you can reopen a trip whenever you want."
      footer={
        <>
          New here?{' '}
          <Link to="/signup" className="text-sky-300 font-semibold hover:text-sky-200">
            Create an account
          </Link>
        </>
      }
    >
      <FormError message={error} />
      <form onSubmit={onSubmit}>
        <Field label="Email">
          <div className="relative">
            <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className={`${inputClass} pl-10`}
            />
          </div>
        </Field>

        <Field label="Password">
          <div className="relative">
            <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="password"
              required
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className={`${inputClass} pl-10`}
            />
          </div>
        </Field>

        <button
          type="submit"
          disabled={busy || !email || !password}
          className="btn-accent w-full py-3.5 rounded-xl font-bold text-sm tracking-[0.12em] flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
        >
          {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <LogIn className="w-4 h-4" />}
          {busy ? 'SIGNING IN' : 'SIGN IN'}
        </button>
      </form>
    </AuthShell>
  );
}
