import { useEffect, useState } from 'react';
import axios from 'axios';
import { Star, Send, BarChart3 } from 'lucide-react';

import { API_BASE_URL } from '../utils/apiBase';

interface Props {
  destination: string;
  origin?: string;
  tripTitle?: string;
}

export default function TripRating({ destination, origin, tripTitle }: Props) {
  const [rating, setRating] = useState(0);
  const [hover, setHover] = useState(0);
  const [feedback, setFeedback] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [stats, setStats] = useState<{ count: number; avg: number; distribution: Record<string, number> } | null>(null);
  const [recent, setRecent] = useState<any[]>([]);
  const [submitted, setSubmitted] = useState(false);

  const fetchStats = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/api/trip-rating`, { params: { destination } });
      setStats(res.data.stats);
      setRecent(res.data.ratings || []);
    } catch {}
  };

  useEffect(() => { if (destination) fetchStats(); }, [destination]);

  const submit = async () => {
    if (rating < 1) return;
    setSubmitting(true);
    try {
      await axios.post(`${API_BASE_URL}/api/trip-rating`, { destination, rating, feedback, origin, trip_title: tripTitle });
      setSubmitted(true);
      setFeedback('');
      await fetchStats();
      setTimeout(() => setSubmitted(false), 3000);
    } catch (e) { console.error(e); }
    finally { setSubmitting(false); }
  };

  return (
    <div className="glass-card p-6 md:p-8 space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-400/20 to-rose-400/10 flex items-center justify-center border border-amber-400/20">
          <Star className="w-5 h-5 text-amber-400" />
        </div>
        <div>
          <h3 className="text-sm font-bold text-white tracking-widest uppercase">Rate Your Itinerary</h3>
          <p className="text-[11px] text-slate-400">Help improve XPLORA — your feedback fine-tunes future trips (RLHF loop).</p>
        </div>
      </div>

      {stats && stats.count > 0 && (
        <div className="bg-white/[0.04] rounded-xl p-4 border border-white/5 flex items-center gap-6 flex-wrap">
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-white">{stats.avg.toFixed(1)}</span>
            <Star className="w-4 h-4 fill-amber-400 text-amber-400" />
            <span className="text-xs text-slate-400">({stats.count} ratings)</span>
          </div>
          <div className="flex items-center gap-1.5">
            {[5,4,3,2,1].map(s => (
              <span key={s} className="text-[10px] text-slate-500">{s}★ {stats.distribution[String(s)] || 0}</span>
            ))}
          </div>
          <span className="ml-auto text-[10px] text-teal-300 flex items-center gap-1"><BarChart3 className="w-3 h-3" /> Live social proof</span>
        </div>
      )}

      <div className="space-y-3">
        <div className="flex gap-1.5">
          {[1,2,3,4,5].map(n => (
            <button key={n} onClick={() => setRating(n)} onMouseEnter={() => setHover(n)} onMouseLeave={() => setHover(0)}
              className="p-1.5 rounded-lg bg-white/[0.04] border border-white/10 hover:border-amber-400/30 transition-colors">
              <Star className={`w-7 h-7 ${n <= (hover || rating) ? 'fill-amber-400 text-amber-400' : 'text-white/20'}`} />
            </button>
          ))}
        </div>
        <textarea value={feedback} onChange={e=>setFeedback(e.target.value)} placeholder="What did you love or want improved? (optional)"
          className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-sm text-white placeholder:text-slate-500 focus:border-amber-400/30 outline-none" rows={2} />
        <button onClick={submit} disabled={submitting || rating===0}
          className="bg-gradient-to-r from-amber-500 to-rose-500 text-white font-bold px-6 py-2.5 rounded-xl text-xs tracking-widest disabled:opacity-40 flex items-center gap-2 hover:shadow-[0_0_20px_rgba(251,113,133,0.3)] transition-all">
          <Send className="w-4 h-4" /> {submitting ? 'SUBMITTING...' : submitted ? 'THANK YOU!' : 'SUBMIT RATING'}
        </button>
        {submitted && <p className="text-xs text-emerald-400 font-medium">Thanks! Your rating powers the RLHF loop for future itineraries.</p>}
      </div>

      {recent.length > 0 && (
        <div className="space-y-2 pt-2 border-t border-white/5">
          <p className="text-[10px] font-bold tracking-widest text-slate-500 uppercase">Recent Community Ratings</p>
          {recent.slice(0,3).map((r:any,i:number)=>(
            <div key={i} className="bg-white/[0.03] rounded-lg p-3 border border-white/5">
              <div className="flex items-center gap-2">
                <span className="flex">{Array.from({length:5}).map((_,k)=><Star key={k} className={`w-3 h-3 ${k < r.rating ? 'fill-amber-400 text-amber-400' : 'text-white/15'}`} />)}</span>
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
