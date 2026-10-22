import { useMemo, useState, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Loader2, Mail, Lock, Sparkles, User } from 'lucide-react';
import AuthShell, { Field, FormError, inputClass } from '../components/AuthShell';
import { apiError, signUp } from '../utils/auth';

const MIN_PASSWORD = 8;

export default function Signup() {
  const navigate = useNavigate();
  const [displayName, setDisplayName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const passwordIssue = useMemo(() => {
    if (!password) return 'At least 8 characters.';
    if (password.length < MIN_PASSWORD) return `${MIN_PASSWORD - password.length} more character(s) needed.`;
    if (!confirm) return 'Re-type the password to confirm.';
    if (password !== confirm) return 'Passwords do not match.';
    return null;
  }, [password, confirm]);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (busy || passwordIssue) return;
    setBusy(true);
    setError(null);
    try {
      await signUp({ email: email.trim(), password, confirmPassword: confirm, displayName: displayName.trim() });
      navigate('/dashboard', { replace: true });
    } catch (err) {
      setError(apiError(err));
      setBusy(false);
    }
  };

  return (
    <AuthShell
      eyebrow="Get started"
      title="Create your account"
      subtitle="Sign up once and every itinerary the agents build for you is saved to your own dashboard."
      footer={
        <>
          Already have an account?{' '}
          <Link to="/login" className="text-sky-300 font-semibold hover:text-sky-200">
            Sign in
          </Link>
        </>
      }
    >
      <FormError message={error} />
      <form onSubmit={onSubmit}>
        <Field label="Name (Optional)">
          <div className="relative">
            <User className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="text"
              maxLength={80}
              autoComplete="name"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="What should we call you?"
              className={`${inputClass} pl-10`}
            />
          </div>
        </Field>

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

        <Field label="Password" hint={password && !passwordIssue ? 'Strong enough.' : 'Use at least 8 characters.'}>
          <div className="relative">
            <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="password"
              required
              minLength={MIN_PASSWORD}
              autoComplete="new-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className={`${inputClass} pl-10`}
            />
          </div>
        </Field>

        <Field label="Confirm Password">
          <div className="relative">
            <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="password"
              required
              minLength={MIN_PASSWORD}
              autoComplete="new-password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              placeholder="••••••••"
              className={`${inputClass} pl-10`}
            />
          </div>
        </Field>

        <button
          type="submit"
          disabled={busy || !email || !!passwordIssue}
          className="btn-accent w-full py-3.5 rounded-xl font-bold text-sm tracking-[0.12em] flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
        >
          {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
          {busy ? 'CREATING ACCOUNT' : 'CREATE ACCOUNT'}
        </button>
      </form>
    </AuthShell>
  );
}
