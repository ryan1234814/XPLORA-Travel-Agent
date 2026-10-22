import { useCallback, useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  CalendarDays, Compass, Loader2, LogOut, MapPin, Plus, Trash2, Users, Wallet,
} from 'lucide-react';
import BrandMark from '../components/BrandMark';
import {
  apiError, deleteSavedItinerary, fetchSavedItineraries, signOut, useAuth,
  type SavedItinerarySummary,
} from '../utils/auth';
import { describeDuration, formatSavedOn } from '../utils/savedTrips';

function TripCard({
  trip,
  onOpen,
  onDelete,
  index,
}: {
  trip: SavedItinerarySummary;
  onOpen: (id: number) => void;
  onDelete: (id: number) => void;
  index: number;
}) {
  const headline = trip.title || (trip.destination ? `Trip to ${trip.destination}` : 'Saved itinerary');
  const chips = [
    trip.travel_dates,
    describeDuration(trip.duration),
    trip.budget,
    trip.group_type ? `${trip.group_type}${trip.group_size ? ` · ${trip.group_size}` : ''}` : null,
    trip.pace,
    ...(Array.isArray(trip.interests) ? trip.interests.slice(0, 2) : []),
  ].filter((c): c is string => !!c && String(c).trim().length > 0);

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: Math.min(index * 0.04, 0.3), ease: [0.16, 1, 0.3, 1] }}
      className="flat-card p-5 flex flex-col gap-3 group"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-[9px] font-bold text-sky-300 uppercase tracking-[0.25em] mb-1.5 flex items-center gap-1.5">
            <MapPin className="w-3 h-3" />
            {trip.destination || 'Unknown destination'}
          </p>
          <h3 className="text-lg font-bold text-white leading-snug truncate" title={headline}>{headline}</h3>
        </div>
        <button
          onClick={() => onDelete(trip.id)}
          title="Delete this saved trip"
          className="shrink-0 p-2 rounded-lg border border-white/10 text-slate-500 opacity-0 group-hover:opacity-100 hover:text-red-300 hover:border-red-400/40 transition-all"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {chips.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {chips.map((c) => (
            <span
              key={c}
              className="text-[10px] px-2.5 py-1 rounded-full border border-white/10 bg-white/[0.05] text-slate-300"
            >
              {c}
            </span>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between gap-3 mt-auto pt-1">
        <p className="text-[10px] text-slate-500 flex items-center gap-1.5">
          <CalendarDays className="w-3 h-3" />
          Saved {formatSavedOn(trip.created_at) || 'recently'}
          {trip.origin ? (
            <>
              <span className="text-slate-700">·</span>
              <Users className="w-3 h-3" /> from {trip.origin}
            </>
          ) : null}
        </p>
        <button
          onClick={() => onOpen(trip.id)}
          className="text-[10px] font-bold uppercase tracking-[0.18em] px-3.5 py-2 rounded-xl border border-sky-400/40 text-sky-300 bg-sky-400/[0.06] hover:bg-sky-400/[0.14] hover:text-white transition-all flex items-center gap-1.5"
        >
          <Compass className="w-3.5 h-3.5" />
          Open trip
        </button>
      </div>
    </motion.div>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [trips, setTrips] = useState<SavedItinerarySummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setTrips(await fetchSavedItineraries());
    } catch (err) {
      setError(apiError(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const onDelete = async (id: number) => {
    if (!window.confirm('Delete this saved trip? The itinerary will be removed from your account.')) return;
    // Optimistic removal; restored below if the request fails.
    const previous = trips ?? [];
    setTrips(previous.filter((t) => t.id !== id));
    try {
      await deleteSavedItinerary(id);
    } catch (err) {
      setTrips(previous);
      setError(apiError(err));
    }
  };

  const totalDays = (trips ?? []).reduce((sum, t) => sum + (t.duration || 0), 0);
  const destinations = new Set((trips ?? []).map((t) => (t.destination || '').toLowerCase()).filter(Boolean));

  return (
    <div className="main-gradient min-h-screen text-slate-200">
      <header className="sticky top-0 z-30 bg-[#0c0e12]/90 backdrop-blur-md border-b border-white/5 relative">
        <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-sky-400 via-sky-400 via-sky-400 to-slate-500 opacity-60"></div>
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between gap-4">
          <Link to="/" className="flex items-center gap-3 text-left group">
            <div className="bg-gradient-to-br from-sky-400/25 via-sky-400/15 to-sky-400/20 p-2.5 rounded-2xl border border-sky-400/10 shadow-[0_0_30px_rgba(56,189,248,0.2)] transition-transform duration-300 group-hover:scale-105">
              <BrandMark className="w-6 h-6 text-sky-300 drop-shadow-[0_0_8px_rgba(56,189,248,0.5)]" />
            </div>
            <div>
              <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-white leading-tight">
                <span className="wordmark">XPLORA</span>
              </h1>
              <p className="text-[9px] sm:text-[10px] text-slate-500 italic tracking-[0.15em]">Your travel dashboard</p>
            </div>
          </Link>

          <div className="flex items-center gap-2 shrink-0">
            <span className="hidden sm:block text-[10px] uppercase tracking-[0.2em] text-slate-500 mr-1 truncate max-w-[180px]">
              {user?.display_name || user?.email}
            </span>
            <button
              onClick={() => navigate('/app')}
              className="btn-accent px-3 sm:px-4 py-2 rounded-xl text-[10px] sm:text-xs font-bold uppercase tracking-widest flex items-center gap-1.5"
            >
              <Plus className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Plan a trip</span>
              <span className="sm:hidden">Plan</span>
            </button>
            <button
              onClick={async () => {
                await signOut();
                navigate('/login', { replace: true });
              }}
              title="Sign out"
              className="p-2 rounded-xl border border-white/10 text-slate-400 hover:text-white hover:bg-white/[0.06] transition-all"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-8 sm:py-10">
        <div className="mb-8">
          <p className="text-[10px] font-bold text-sky-300 uppercase tracking-[0.3em] mb-3">Your journeys</p>
          <h2 className="text-3xl sm:text-4xl font-bold text-white leading-tight">
            {user?.display_name ? `${user.display_name.split(' ')[0]}, your` : 'Your'} saved itineraries
          </h2>
          <p className="text-sm text-slate-400 font-light mt-2">
            Every plan the agents generated while you were signed in lives here. Open one to pick up exactly where you left off.
          </p>
        </div>

        {!loading && trips && trips.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {[
              { label: 'Trips saved', value: trips.length, Icon: Compass },
              { label: 'Destinations', value: destinations.size, Icon: MapPin },
              { label: 'Days planned', value: totalDays, Icon: CalendarDays },
              { label: 'Latest', value: formatSavedOn(trips[0].created_at) || '—', Icon: Wallet, small: true },
            ].map((s) => (
              <div key={s.label} className="bg-[#0c0e12]/70 border border-white/10 rounded-2xl p-4">
                <p className="text-[9px] uppercase tracking-[0.25em] text-slate-500 mb-2 flex items-center gap-1.5">
                  <s.Icon className="w-3 h-3 text-sky-300" />
                  {s.label}
                </p>
                <p className={`font-bold text-white ${s.small ? 'text-base' : 'text-3xl'}`}>{s.value}</p>
              </div>
            ))}
          </div>
        )}

        {error && (
          <div className="rounded-2xl border border-red-500/30 bg-red-500/[0.07] px-4 py-3 text-xs text-red-300 mb-6 flex items-center justify-between gap-3">
            <span>{error}</span>
            <button onClick={load} className="font-bold uppercase tracking-widest text-[10px] hover:text-white">Retry</button>
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center gap-2 py-20 text-slate-500">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span className="text-[10px] font-bold uppercase tracking-[0.3em]">Loading your trips</span>
          </div>
        ) : !trips || trips.length === 0 ? (
          <div className="flat-card p-10 text-center">
            <div className="mx-auto w-14 h-14 rounded-2xl border border-sky-400/20 bg-sky-400/10 flex items-center justify-center text-sky-300 mb-5">
              <Compass className="w-7 h-7" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">No saved trips yet</h3>
            <p className="text-sm text-slate-400 font-light max-w-md mx-auto mb-6">
              Plan your first journey and it will be archived here automatically — day-by-day schedule, costs,
              transport and local insights included.
            </p>
            <button
              onClick={() => navigate('/app')}
              className="btn-accent px-8 py-3.5 rounded-xl font-bold text-sm tracking-[0.12em] inline-flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              PLAN YOUR FIRST TRIP
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {trips.map((trip, i) => (
              <TripCard
                key={trip.id}
                trip={trip}
                index={i}
                onOpen={(id) => navigate(`/app?trip=${id}`)}
                onDelete={onDelete}
              />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
