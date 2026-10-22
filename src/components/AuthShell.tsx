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
    <div className="main-gradient min-h-screen text-slate-200 flex flex-col items-center justify-center px-4 py-10 relative">
      <div className="absolute inset-x-0 top-0 h-[1px] bg-gradient-to-r from-transparent via-sky-400/40 to-transparent" />

      <button
        onClick={() => navigate('/')}
        className="absolute top-5 left-5 sm:top-7 sm:left-8 flex items-center gap-2 text-[10px] sm:text-xs font-bold uppercase tracking-[0.2em] text-slate-400 hover:text-sky-300 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        Home
      </button>

      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        className="w-full max-w-md"
      >
        <div className="flex flex-col items-center mb-8">
          <div className="bg-gradient-to-br from-sky-400/25 via-sky-400/15 to-sky-400/20 p-4 rounded-2xl border border-sky-400/15 shadow-[0_0_40px_rgba(56,189,248,0.2)] mb-4">
            <BrandMark className="w-9 h-9 text-sky-300 drop-shadow-[0_0_10px_rgba(56,189,248,0.5)]" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-white leading-tight">
            <span className="wordmark">XPLORA</span>
          </h1>
          <p className="text-[9px] text-slate-500 italic tracking-[0.2em] mt-1">Intelligent Travel Architect</p>
        </div>

        <div className="bg-[#0c0e12]/80 border border-white/10 rounded-2xl p-6 sm:p-8 relative">
          <div className="absolute top-0 left-6 right-6 h-[1px] bg-gradient-to-r from-transparent via-sky-400/30 to-transparent" />
          <p className="text-[10px] font-bold text-sky-300 uppercase tracking-[0.3em] mb-2">{eyebrow}</p>
          <h2 className="text-2xl font-bold text-white mb-1.5 leading-snug">{title}</h2>
          <p className="text-sm text-slate-400 font-light mb-6 leading-relaxed">{subtitle}</p>
          {children}
        </div>

        <div className="text-center text-xs text-slate-500 mt-5">{footer}</div>
      </motion.div>
    </div>
  );
}

export function FormError({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div className="flex items-start gap-2 rounded-xl border border-red-500/30 bg-red-500/[0.07] px-3.5 py-3 text-xs text-red-300 mb-5">
      <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
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
    <div className="space-y-2 mb-5">
      <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">{label}</label>
      {children}
      {hint && <p className="text-[10px] text-slate-600 ml-1">{hint}</p>}
    </div>
  );
}

export const inputClass =
  'w-full bg-white/5 border border-white/10 rounded-xl py-3.5 px-4 text-sm outline-none transition-all duration-300 focus:border-sky-400/50 focus:bg-sky-400/[0.06] focus:shadow-[0_0_20px_rgba(56,189,248,0.06)]';
