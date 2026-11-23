import { useEffect, useState } from 'react';
import axios from 'axios';
import { Star, Send, BarChart3 } from 'lucide-react';

import { API_BASE_URL } from '../utils/apiBase';

interface Props {
  destination: string;
  origin?: string;
  tripTitle?: string;
}

interface RecentRating {
  rating: number;
  feedback?: string;
  created_at: string;
}

export default function TripRating({ destination, origin, tripTitle }: Props) {
  const [rating, setRating] = useState(0);
  const [hover, setHover] = useState(0);
  const [feedback, setFeedback] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [stats, setStats] = useState<{ count: number; avg: number; distribution: Record<string, number> } | null>(null);
  const [recent, setRecent] = useState<RecentRating[]>([]);
  const [submitted, setSubmitted] = useState(false);

  const fetchStats = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/api/trip-rating`, { params: { destination }, timeout: 60000 });
      setStats(res.data.stats);
      setRecent(res.data.ratings || []);
    } catch { /* stats are best-effort; ignore fetch failures */ }
  };

  // fetchStats closes over `destination`; re-running on destination change is
  // the intended refetch, so we intentionally omit it from the deps array.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { if (destination) fetchStats(); }, [destination]);

  const submit = async () => {
    if (rating < 1) return;
    setSubmitting(true);
    try {
      await axios.post(`${API_BASE_URL}/api/trip-rating`, { destination, rating, feedback, origin, trip_title: tripTitle }, { timeout: 60000, withCredentials: true });
      setSubmitted(true);
      setFeedback('');
      await fetchStats();
      setTimeout(() => setSubmitted(false), 3000);
    } catch (e) { console.error(e); }
    finally { setSubmitting(false); }
  };

  return (
    <div className="flat-card space-y-5 p-6">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/[0.04]">
          <Star className="h-4 w-4 text-sky-300" />
        </div>
        <div>
          <h3 className="text-[13px] font-medium text-white">Rate this itinerary</h3>
          <p className="text-[11px] text-slate-500">Your feedback fine-tunes future trips.</p>
        </div>
      </div>

      {stats && stats.count > 0 && (
        <div className="flex flex-wrap items-center gap-5 rounded-xl border border-white/[0.07] bg-white/[0.03] p-3.5">
          <div className="flex items-center gap-2">
            <span className="text-[22px] font-semibold text-white">{stats.avg.toFixed(1)}</span>
            <Star className="w-4 h-4 fill-sky-400 text-sky-300" />
            <span className="text-xs text-slate-400">({stats.count} ratings)</span>
          </div>
          <div className="flex items-center gap-1.5">
            {[5,4,3,2,1].map(s => (
              <span key={s} className="text-[10px] text-slate-500">{s}★ {stats.distribution[String(s)] || 0}</span>
            ))}
          </div>
          <span className="ml-auto flex items-center gap-1 text-[10px] text-slate-500"><BarChart3 className="w-3 h-3" /> Live social proof</span>
        </div>
      )}

      <div className="space-y-3">
        <div className="flex gap-1.5">
          {[1,2,3,4,5].map(n => (
            <button key={n} onClick={() => setRating(n)} onMouseEnter={() => setHover(n)} onMouseLeave={() => setHover(0)}
              className="rounded-lg border border-white/[0.08] bg-white/[0.03] p-1.5 transition-colors hover:border-white/20">
              <Star className={`w-7 h-7 ${n <= (hover || rating) ? 'fill-sky-400 text-sky-300' : 'text-slate-700'}`} />
            </button>
          ))}
        </div>
        <textarea value={feedback} onChange={e=>setFeedback(e.target.value)} placeholder="What did you love or want improved? (optional)"
          className="field w-full p-3 text-[13px] outline-none" rows={2} />
        <button onClick={submit} disabled={submitting || rating===0}
          className="btn-accent flex items-center gap-2 rounded-lg px-5 py-2.5 text-[12px] font-medium disabled:opacity-40">
          <Send className="w-4 h-4" /> {submitting ? 'Submitting…' : submitted ? 'Thank you' : 'Submit rating'}
        </button>
        {submitted && <p className="text-xs text-slate-300 font-medium">Thanks! Your rating powers the RLHF loop for future itineraries.</p>}
      </div>

      {recent.length > 0 && (
        <div className="space-y-2 pt-2 border-t border-white/5">
          <p className="text-[10px] font-medium uppercase tracking-[0.12em] text-slate-500">Recent ratings</p>
          {recent.slice(0,3).map((r:RecentRating,i:number)=>(
            <div key={i} className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
              <div className="flex items-center gap-2">
                <span className="flex">{Array.from({length:5}).map((_,k)=><Star key={k} className={`w-3 h-3 ${k < r.rating ? 'fill-sky-400 text-sky-300' : 'text-slate-700'}`} />)}</span>
                <span className="text-[11px] text-slate-300">{r.feedback || '—'}</span>
                <span className="text-[10px] text-slate-500 ml-auto">{new Date(r.created_at).toLocaleDateString()}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
