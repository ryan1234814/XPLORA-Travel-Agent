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
      className="flat-card group flex flex-col gap-3 p-5"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="mb-1.5 flex items-center gap-1.5 text-[10px] uppercase tracking-[0.12em] text-sky-300">
            <MapPin className="h-3 w-3" />
            {trip.destination || 'Unknown destination'}
          </p>
          <h3 className="truncate text-[16px] font-medium leading-snug text-white" title={headline}>{headline}</h3>
        </div>
        <button
          onClick={() => onDelete(trip.id)}
          title="Delete this saved trip"
          className="shrink-0 rounded-lg border border-white/10 p-2 text-slate-500 opacity-0 transition-all hover:border-red-400/40 hover:text-red-300 group-hover:opacity-100"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {chips.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {chips.map((c) => (
            <span
              key={c}
              className="rounded-full border border-white/10 bg-white/[0.05] px-2.5 py-0.5 text-[10px] text-slate-300"
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
          className="flex items-center gap-1.5 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-[11px] font-medium text-sky-200 transition-colors hover:bg-sky-500/20"
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
      <header className="sticky top-0 z-30 border-b border-white/[0.08] bg-black/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-wide items-center justify-between gap-4 px-4 py-2.5 sm:px-6">
          <Link to="/" className="group flex items-center gap-2.5 text-left">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/[0.05] transition-colors duration-300 group-hover:border-sky-500/40">
              <BrandMark className="h-4 w-4 text-sky-300" />
            </span>
            <span className="flex flex-col">
              <span className="text-[15px] font-semibold leading-tight tracking-tight text-white">XPLORA</span>
              <span className="text-[10px] leading-tight text-slate-500">Your travel dashboard</span>
            </span>
          </Link>

          <div className="flex shrink-0 items-center gap-2">
            <span className="mr-1 hidden max-w-[180px] truncate text-[11px] text-slate-500 sm:block">
              {user?.display_name || user?.email}
            </span>
            <button
              onClick={() => navigate('/app')}
              className="btn-accent flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-[11px] font-medium"
            >
              <Plus className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Plan a trip</span>
              <span className="sm:hidden">Plan</span>
            </button>
            <button
              onClick={async () => {
                await signOut();
                navigate('/login', { replace: true });
              }}
              title="Sign out"
              className="rounded-full border border-white/10 p-2 text-slate-400 transition-colors hover:bg-white/[0.06] hover:text-white"
            >
              <LogOut className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-wide px-4 py-10 sm:px-6 sm:py-14">
        <div className="mb-9 max-w-2xl">
          <p className="eyebrow mb-3">
            {user?.display_name ? `Welcome back, ${user.display_name.split(' ')[0]}` : 'Your journeys'}
          </p>
          <h2 className="text-[clamp(1.8rem,4vw,2.75rem)] font-semibold leading-[1.08] tracking-[-0.03em] text-white">
            Your saved itineraries
          </h2>
          <p className="mt-3 text-[14px] leading-relaxed text-slate-400">
            Every plan the agents generated while you were signed in lives here. Open one to pick up exactly where you left off.
          </p>
        </div>

        {!loading && trips && trips.length > 0 && (
          <div className="mb-9 grid grid-cols-2 gap-3 md:grid-cols-4">
            {[
              { label: 'Trips saved', value: trips.length, Icon: Compass },
              { label: 'Destinations', value: destinations.size, Icon: MapPin },
              { label: 'Days planned', value: totalDays, Icon: CalendarDays },
              { label: 'Latest', value: formatSavedOn(trips[0].created_at) || '—', Icon: Wallet, small: true },
            ].map((s) => (
              <div key={s.label} className="rounded-xl border border-white/[0.08] bg-white/[0.03] p-4">
                <p className="mb-2 flex items-center gap-1.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
                  <s.Icon className="h-3 w-3 text-sky-300" />
                  {s.label}
                </p>
                <p className={`font-semibold text-white ${s.small ? 'text-[15px]' : 'text-[26px] tracking-tight'}`}>{s.value}</p>
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
          <div className="flex items-center justify-center gap-2.5 py-24 text-slate-500">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-[11px]">Loading your trips…</span>
          </div>
        ) : !trips || trips.length === 0 ? (
          <div className="flat-card p-10 text-center">
            <div className="mx-auto mb-5 flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.04] text-sky-300">
              <Compass className="h-6 w-6" />
            </div>
            <h3 className="mb-2 text-[19px] font-semibold text-white">No saved trips yet</h3>
            <p className="mx-auto mb-6 max-w-md text-[13.5px] leading-relaxed text-slate-400">
              Plan your first journey and it will be archived here automatically — day-by-day schedule, costs,
              transport and local insights included.
            </p>
            <button
              onClick={() => navigate('/app')}
              className="btn-accent inline-flex items-center gap-2 rounded-full px-6 py-3 text-[13px] font-medium"
            >
              <Plus className="h-4 w-4" />
              Plan your first trip
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-3.5 md:grid-cols-2 lg:grid-cols-3">
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
