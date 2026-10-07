import { useEffect, useMemo, useState } from 'react';
import { Check, Luggage, Receipt, Sun, Thermometer } from 'lucide-react';
import type { WeatherData } from '../App';
import {
  BUDGET_TIERS,
  COST_CATEGORY_LABELS,
  COST_TIERS,
  formatLocalNumber,
  formatPriceRange,
  getCurrencyForDestination,
  getExchangeMultiplier,
  isBudgetTier,
} from '../utils/currency';
import type { BudgetTier, CostCategory } from '../utils/currency';
import { buildPackList } from '../utils/pack';
import type { PackCategory, PackItem } from '../utils/pack';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SmartPackTrackerProps {
  weather: WeatherData | null;
  destination: string;
  budget: string;
  duration: number;
  groupSize: number;
  pace: string;
  interests: string[];
}

const PACK_CATEGORY_ORDER: PackCategory[] = ['Essentials', 'Clothing', 'Tech/Health', 'Docs'];

// ---------------------------------------------------------------------------
// localStorage helpers (all guarded so the component stays safe offline / SSR)
// ---------------------------------------------------------------------------

function safeGet(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeSet(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* storage unavailable — feature still works in-memory */
  }
}

function readStringArray(key: string): string[] {
  try {
    const raw = safeGet(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((v): v is string => typeof v === 'string') : [];
  } catch {
    return [];
  }
}

function destinationKey(destination: string, prefix: string): string {
  return `${prefix}${(destination || 'destination').trim().toLowerCase() || 'destination'}`;
}

// ---------------------------------------------------------------------------
// Category render metadata
// ---------------------------------------------------------------------------

const CATEGORY_META: Record<PackCategory, { accent: string; dot: string }> = {
  Essentials: { accent: 'text-sky-300', dot: 'bg-sky-400' },
  Clothing: { accent: 'text-sky-300', dot: 'bg-sky-400' },
  'Tech/Health': { accent: 'text-sky-300', dot: 'bg-sky-400' },
  Docs: { accent: 'text-slate-300', dot: 'bg-slate-700' },
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function SmartPackTracker({
  weather,
  destination,
  budget,
  duration,
  groupSize,
  pace,
  interests,
}: SmartPackTrackerProps) {
  // ---- FX / budget numbers ------------------------------------------------
  const [symbol, code, name] = useMemo(() => getCurrencyForDestination(destination), [destination]);
  const multiplier = useMemo(() => getExchangeMultiplier(code), [code]);
  const tier = useMemo<BudgetTier>(() => (isBudgetTier(budget) ? budget : 'Premier'), [budget]);
  const tierBounds = BUDGET_TIERS[tier];
  const costTiers = COST_TIERS[tier];

  const totalHigh = Math.round(tierBounds[1] * duration * multiplier);
  const perDayLow = Math.round(tierBounds[0] * multiplier);
  const perDayHigh = Math.round(tierBounds[1] * multiplier);

  const fmt = (n: number) => `${symbol}${formatLocalNumber(n)}`;
  const priceRange = useMemo(
    () => formatPriceRange(destination, budget, duration),
    [destination, budget, duration],
  );

  // ---- Derived pack list (LLM packing + weather/interest rules) -----------
  const typicalRange = weather?.temperature_c?.typical_range || '—';

  const packItems = useMemo<PackItem[]>(
    () => buildPackList(weather, { pace, interests }),
    [weather, pace, interests],
  );

  // ---- Pack checklist state (persisted per destination) --------------------
  // The parent renders this component with `key={destination}`, so this state is
  // re-initialized from localStorage whenever the destination changes.
  const packKey = useMemo(() => destinationKey(destination, 'xplora-pack-'), [destination]);
  const [checked, setChecked] = useState<string[]>(() => {
    const saved = readStringArray(packKey);
    return packItems.filter((item) => saved.includes(item.id)).map((item) => item.id);
  });

  useEffect(() => {
    safeSet(packKey, JSON.stringify(checked));
  }, [checked, packKey]);

  const toggleItem = (id: string) => {
    setChecked((prev) => (prev.includes(id) ? prev.filter((c) => c !== id) : [...prev, id]));
  };

  const packedCount = packItems.filter((item) => checked.includes(item.id)).length;
  const packedPct = packItems.length > 0 ? Math.round((packedCount / packItems.length) * 100) : 0;

  // ---- Spent tracker state (persisted per destination) ---------------------
  const spentKey = useMemo(() => destinationKey(destination, 'xplora-spent-'), [destination]);
  const [spentText, setSpentText] = useState<string>(() => safeGet(spentKey) ?? '');

  useEffect(() => {
    safeSet(spentKey, spentText);
  }, [spentText, spentKey]);

  const spentNum = Number(spentText);
  const spent = Number.isFinite(spentNum) && spentNum > 0 ? spentNum : 0;
  const remaining = totalHigh - spent;
  const spentPct = totalHigh > 0 ? Math.min(100, Math.round((spent / totalHigh) * 100)) : 0;
  const barColor =
    spentPct < 60
      ? 'bg-gradient-to-r from-slate-500 to-sky-400'
      : spentPct <= 85
        ? 'bg-gradient-to-r from-sky-400 to-orange-400'
        : 'bg-gradient-to-r from-slate-500 to-slate-500';
  const statusLabel = remaining >= 0 ? 'On track' : 'Over budget';

  // ---- Grouped categories for rendering ------------------------------------
  const grouped = useMemo(() => {
    const groups = new Map<PackCategory, PackItem[]>();
    for (const item of packItems) {
      const list = groups.get(item.category) ?? [];
      list.push(item);
      groups.set(item.category, list);
    }
    return groups;
  }, [packItems]);

  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
      {/* ============================== SMART PACK ============================== */}
      <div className="flat-card p-6 relative overflow-hidden">
        <div className="absolute -top-12 -right-12 w-36 h-36 bg-sky-400/10 blur-3xl rounded-full pointer-events-none"></div>
        <div className="flex items-start justify-between gap-4 mb-5 relative z-10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-400/25 to-sky-400/10 border border-sky-400/20 flex items-center justify-center shrink-0">
              <Luggage className="w-5 h-5 text-sky-300" />
            </div>
            <div>
              <h3 className="text-white font-bold tracking-wide leading-tight">Smart Pack</h3>
              <p className="text-[11px] text-slate-400 mt-0.5 leading-snug">
                Based on {typicalRange}
                {weather?.conditions_summary ? ` • ${weather.conditions_summary}` : ''}
              </p>
            </div>
          </div>
          <div className="text-right shrink-0">
            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Packed</div>
            <div className="text-sm font-bold text-white">
              {packedCount}
              <span className="text-slate-500">/{packItems.length}</span>
            </div>
          </div>
        </div>

        {/* Progress bar */}
        <div className="h-1.5 rounded-full bg-white/[0.06] overflow-hidden mb-5 relative z-10">
          <div
            className="h-full rounded-full bg-gradient-to-r from-sky-400 via-sky-400 to-sky-400 transition-all duration-500"
            style={{ width: `${packedPct}%` }}
          ></div>
        </div>

        {/* Grouped checklist */}
        <div className="space-y-5 relative z-10 max-h-[340px] overflow-y-auto pr-1 custom-scrollbar">
          {PACK_CATEGORY_ORDER.filter((cat) => (grouped.get(cat)?.length ?? 0) > 0).map((cat) => (
            <div key={cat}>
              <div className={`text-[10px] font-bold uppercase tracking-widest mb-2 flex items-center gap-2 ${CATEGORY_META[cat].accent}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${CATEGORY_META[cat].dot}`}></span>
                {cat}
              </div>
              <div className="space-y-1.5">
                {grouped.get(cat)?.map((item) => {
                  const isChecked = checked.includes(item.id);
                  return (
                    <label
                      key={item.id}
                      className={`flex items-start gap-3 px-3 py-2 rounded-xl border transition-all duration-300 cursor-pointer group/row ${
                        isChecked
                          ? 'bg-sky-400/[0.06] border-sky-400/20'
                          : 'bg-white/[0.02] border-white/[0.05] hover:bg-white/[0.05] hover:border-white/10'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => toggleItem(item.id)}
                        className="mt-0.5 w-4 h-4 rounded accent-primary cursor-pointer shrink-0"
                      />
                      <span
                        className={`text-xs leading-relaxed transition-colors duration-300 ${
                          isChecked ? 'text-slate-500 line-through' : 'text-slate-200 group-hover/row:text-white'
                        }`}
                      >
                        {item.label}
                      </span>
                      {isChecked && <Check className="w-3.5 h-3.5 text-sky-300 ml-auto shrink-0 mt-0.5" />}
                    </label>
                  );
                })}
              </div>
            </div>
          ))}
          {packItems.length === 0 && (
            <p className="text-xs text-slate-500 italic">Weather intelligence is still syncing — pack smart essentials first.</p>
          )}
        </div>

        {/* Footer chips */}
        <div className="mt-5 pt-4 border-t border-white/[0.06] flex flex-wrap gap-2 relative z-10">
          <span className="inline-flex items-center gap-1.5 text-[10px] font-medium text-slate-400 bg-white/[0.04] border border-white/[0.06] rounded-full px-3 py-1.5">
            <Sun className="w-3 h-3 text-sky-300" />
            Best times: {(weather?.best_times && weather.best_times.length > 0 ? weather.best_times.join(' • ') : '—')}
          </span>
          <span className="inline-flex items-center gap-1.5 text-[10px] font-medium text-slate-400 bg-white/[0.04] border border-white/[0.06] rounded-full px-3 py-1.5">
            <Thermometer className="w-3 h-3 text-slate-300" />
            Expected: {typicalRange}
          </span>
        </div>
      </div>

      {/* ============================ FX BUDGET TRACKER ============================ */}
      <div className="flat-card p-6 relative overflow-hidden">
        <div className="absolute -bottom-12 -left-12 w-36 h-36 bg-sky-400/10 blur-3xl rounded-full pointer-events-none"></div>
        <div className="flex items-start justify-between gap-4 mb-5 relative z-10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-400/25 to-slate-500/10 border border-sky-400/20 flex items-center justify-center shrink-0">
              <Receipt className="w-5 h-5 text-sky-300" />
            </div>
            <div>
              <h3 className="text-white font-bold tracking-wide leading-tight">Budget Tracker</h3>
              <p className="text-[11px] text-slate-400 mt-0.5 leading-snug">{name}</p>
            </div>
          </div>
          <span className="inline-flex items-center gap-1.5 text-[10px] font-bold text-sky-300 bg-sky-400/10 border border-sky-400/25 rounded-full px-3 py-1.5 whitespace-nowrap shrink-0">
            Tier: {tier}
          </span>
        </div>

        {/* Trip totals */}
        <div className="grid grid-cols-3 gap-2 mb-4 relative z-10">
          <div className="bg-white/[0.03] border border-white/[0.05] rounded-xl px-3 py-2.5">
            <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Total Est</div>
            <div className="text-[11px] font-bold text-white leading-tight whitespace-nowrap overflow-hidden text-ellipsis">{priceRange}</div>
          </div>
          <div className="bg-white/[0.03] border border-white/[0.05] rounded-xl px-3 py-2.5">
            <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Per Day</div>
            <div className="text-[11px] font-bold text-white leading-tight whitespace-nowrap overflow-hidden text-ellipsis">
              {fmt(perDayLow)} - {fmt(perDayHigh)}
            </div>
          </div>
          <div className="bg-white/[0.03] border border-white/[0.05] rounded-xl px-3 py-2.5">
            <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Per Person/Day</div>
            <div className="text-[11px] font-bold text-white leading-tight whitespace-nowrap overflow-hidden text-ellipsis">
              {fmt(perDayLow)} - {fmt(perDayHigh)}
            </div>
          </div>
        </div>

        {/* Per-category daily ranges (in local currency) */}
        <div className="space-y-1 mb-5 relative z-10">
          {(Object.keys(COST_CATEGORY_LABELS) as CostCategory[]).map((cat, idx) => {
            const [catLow, catHigh] = costTiers[cat];
            const dotColors = ['bg-sky-400', 'bg-slate-600', 'bg-sky-400', 'bg-sky-400', 'bg-slate-700', 'bg-slate-700', 'bg-slate-700'];
            return (
              <div
                key={cat}
                className="flex items-center justify-between px-3 py-1.5 rounded-lg bg-white/[0.02] border border-white/[0.03] hover:bg-white/[0.04] hover:border-white/[0.08] transition-all duration-300"
              >
                <span className="flex items-center gap-2 text-[11px] text-slate-400 font-medium">
                  <span className={`w-1.5 h-1.5 rounded-full ${dotColors[idx % dotColors.length]}`}></span>
                  {COST_CATEGORY_LABELS[cat]}
                </span>
                <span className="text-[11px] font-bold text-slate-100 tabular-nums">
                  {fmt(Math.round(catLow * multiplier))} - {fmt(Math.round(catHigh * multiplier))}
                </span>
              </div>
            );
          })}
        </div>

        {/* Spent tracker */}
        <div className="relative z-10 bg-white/[0.03] border border-white/[0.06] rounded-2xl p-4">
          <div className="flex items-center justify-between mb-3">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Spent so far ({code})</label>
            <span
              className={`text-[10px] font-bold uppercase tracking-widest px-2.5 py-1 rounded-full ${
                remaining >= 0 ? 'text-slate-300 bg-slate-600/10 border border-white/20' : 'text-slate-300 bg-slate-700/10 border border-white/20'
              }`}
            >
              {statusLabel}
            </span>
          </div>
          <div className="flex items-center gap-2 mb-3">
            <span className="text-sm text-slate-400 font-semibold">{symbol}</span>
            <input
              type="number"
              min={0}
              step={multiplier < 1 ? 0.1 : 1}
              inputMode="decimal"
              value={spentText}
              placeholder="0"
              onChange={(e) => setSpentText(e.target.value)}
              className="w-full bg-[#0c0e12] border border-white/10 rounded-xl px-3 py-2 text-sm font-bold text-white focus:border-sky-400/50 focus:outline-none transition-all duration-300 tabular-nums"
            />
          </div>
          <input
            type="range"
            min={0}
            max={Math.max(totalHigh, 1)}
            step={multiplier < 1 ? 0.1 : 1}
            value={Math.min(spent, totalHigh)}
            onChange={(e) => setSpentText(e.target.value)}
            className="w-full accent-teal-400 cursor-pointer"
            aria-label="Amount spent so far"
          />
          <div className="flex items-center justify-between mt-3 gap-3">
            <div>
              <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-0.5">Remaining</div>
              <div className={`text-sm font-bold tabular-nums ${remaining >= 0 ? 'text-white' : 'text-slate-300'}`}>
                {remaining < 0 ? '-' : ''}
                {fmt(Math.abs(remaining))}
              </div>
            </div>
            {groupSize > 1 && (
              <div className="text-right">
                <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-0.5">Group total ({groupSize})</div>
                <div className="text-sm font-bold text-sky-300 tabular-nums">{fmt(Math.round(spent * groupSize))}</div>
              </div>
            )}
          </div>
          <div className="h-1.5 rounded-full bg-white/[0.06] overflow-hidden mt-3">
            <div className={`h-full rounded-full transition-all duration-500 ${barColor}`} style={{ width: `${spentPct}%` }}></div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-4 pt-3 border-t border-white/[0.06] flex flex-wrap items-center justify-between gap-2 relative z-10">
          <span className="text-[10px] text-slate-500 font-medium">
            1 USD ≈ {multiplier} {code}
          </span>
          <span className="text-[10px] text-slate-600 italic">Rates approx, no API</span>
        </div>
      </div>
    </div>
  );
}
