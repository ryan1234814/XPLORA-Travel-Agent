import type { ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, AlertCircle } from 'lucide-react';
import BrandMark from './BrandMark';

interface Props {
  eyebrow: string;
  title: ReactNode;
  subtitle: string;
  children: ReactNode;
  footer: ReactNode;
}

/** Shared chrome for the sign-in and sign-up screens. */
export default function AuthShell({ eyebrow, title, subtitle, children, footer }: Props) {
  const navigate = useNavigate();
  return (
    <div className="relative flex min-h-screen flex-col items-center justify-center bg-black px-5 py-14 text-slate-200">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-x-0 top-0 h-[420px] bg-[radial-gradient(ellipse_55%_60%_at_50%_0%,rgba(26,115,232,0.18),transparent_70%)]"
      />

      <button
        onClick={() => navigate('/')}
        className="absolute left-5 top-5 z-10 flex items-center gap-1.5 text-[12px] text-slate-400 transition-colors hover:text-white sm:left-8 sm:top-6"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Home
      </button>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.28, 0.11, 0.32, 1] }}
        className="relative w-full max-w-[400px]"
      >
        <div className="mb-8 flex flex-col items-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.05]">
            <BrandMark className="h-6 w-6 text-sky-300" />
          </div>
          <h1 className="text-[22px] font-semibold tracking-tight text-white">XPLORA</h1>
          <p className="mt-1 text-[11px] tracking-wide text-slate-500">Intelligent Travel Architect</p>
        </div>

        <div className="rounded-2xl border border-white/[0.09] bg-[#121213] p-6 sm:p-7">
          <p className="eyebrow mb-2">{eyebrow}</p>
          <h2 className="mb-1.5 text-[20px] font-semibold leading-snug text-white">{title}</h2>
          <p className="mb-6 text-[13px] leading-relaxed text-slate-400">{subtitle}</p>
          {children}
        </div>

        <div className="mt-6 text-center text-[12px] text-slate-500">{footer}</div>
      </motion.div>
    </div>
  );
}

export function FormError({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div
      role="alert"
      aria-live="assertive"
      className="mb-5 flex items-start gap-2 rounded-xl border border-red-500/30 bg-red-500/[0.08] px-3.5 py-3 text-[12px] text-red-300"
    >
      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
      <span>{message}</span>
    </div>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="mb-5 space-y-1.5">
      <label className="ml-0.5 block text-[11px] font-medium text-slate-400">{label}</label>
      {children}
      {hint && <p className="ml-0.5 text-[11px] text-slate-600">{hint}</p>}
    </div>
  );
}

export const inputClass =
  'field w-full px-3.5 py-3 text-[14px] outline-none';
