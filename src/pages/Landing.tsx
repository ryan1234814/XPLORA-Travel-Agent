import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import {
  Compass,
  Search,
  Brain,
  Globe,
  Wallet,
  Navigation,
  MapPin,
  Clock,
  Sparkles,
  ArrowRight,
  LayoutDashboard,
  CalendarDays,
  ChevronRight,
  Play,
  Pause,
  Volume2,
  VolumeX,
} from 'lucide-react';
import BrandMark from '../components/BrandMark';
import { useAuth } from '../utils/auth';

/* ------------------------------------------------------------------ *
 * Content
 * ------------------------------------------------------------------ */

type Destination = {
  name: string;
  country: string;
  slug: string;
  blurb: string;
  credit: { author: string; license: string; href: string };
};

/** Real, openly-licensed photography (Wikimedia Commons). */
const DESTINATIONS: Destination[] = [
  {
    name: 'Kyoto',
    country: 'Japan',
    slug: 'kyoto',
    blurb: 'Lantern-lit lanes, temple mornings',
    credit: { author: '663highland', license: 'CC BY 2.5', href: 'https://commons.wikimedia.org/wiki/File:150124_At_Yasakakamimachi_Kyoto_Japan01n.jpg' },
  },
  {
    name: 'Santorini',
    country: 'Greece',
    slug: 'santorini',
    blurb: 'Caldera sunsets and whitewashed cliffs',
    credit: { author: 'Giles Laurent', license: 'CC BY-SA 4.0', href: 'https://commons.wikimedia.org/wiki/File:1000_Three_domes_of_Oia_in_Santorini_Photo_by_Giles_Laurent.jpg' },
  },
  {
    name: 'Patagonia',
    country: 'Chile',
    slug: 'patagonia',
    blurb: 'Granite towers over glacial water',
    credit: { author: 'Winky', license: 'CC BY 2.0', href: 'https://commons.wikimedia.org/wiki/File:A_view_towards_Torres_Del_Paine.jpg' },
  },
  {
    name: 'Tokyo',
    country: 'Japan',
    slug: 'tokyo',
    blurb: 'Neon crossings, quiet shrines',
    credit: { author: 'David Kernan', license: 'CC BY 4.0', href: 'https://commons.wikimedia.org/wiki/File:Shibuya_Crossing,_Aerial.jpg' },
  },
  {
    name: 'Petra',
    country: 'Jordan',
    slug: 'petra',
    blurb: 'A rose-red city carved into rock',
    credit: { author: 'Vyacheslav Argenberg', license: 'CC BY 4.0', href: 'https://commons.wikimedia.org/wiki/File:Al-Khazneh,_Petra,_Jordan.jpg' },
  },
  {
    name: 'Amalfi',
    country: 'Italy',
    slug: 'amalfi',
    blurb: 'Lemon groves above a cobalt sea',
    credit: { author: 'Wiki.Bianco', license: 'CC BY-SA 4.0', href: 'https://commons.wikimedia.org/wiki/File:Amalfi_Coast_(Positano).jpg' },
  },
  {
    name: 'Marrakech',
    country: 'Morocco',
    slug: 'marrakech',
    blurb: 'Spice markets and courtyard riads',
    credit: { author: 'David Broad', license: 'CC BY 3.0', href: 'https://commons.wikimedia.org/wiki/File:Medina_souk,_Marrakech,_Morocco_-_panoramio.jpg' },
  },
  {
    name: 'Lisbon',
    country: 'Portugal',
    slug: 'lisbon',
    blurb: 'Tiled facades and tram-lined hills',
    credit: { author: 'Sharon Hahn Darlin', license: 'CC BY 2.0', href: 'https://commons.wikimedia.org/wiki/File:Alfama,_Lisbon,_Portugal_July_2021_-_Tram.jpg' },
  },
  {
    name: 'Banff',
    country: 'Canada',
    slug: 'banff',
    blurb: 'Turquoise lakes under the Rockies',
    credit: { author: 'Chensiyuan', license: 'CC BY-SA 4.0', href: 'https://commons.wikimedia.org/wiki/File:1_moraine_lake_pano_2019.jpg' },
  },
  {
    name: 'Hanoi',
    country: 'Vietnam',
    slug: 'hanoi',
    blurb: 'Old-quarter bustle, lakeside calm',
    credit: { author: 'Mig Gilbert', license: 'CC BY-SA 2.0', href: 'https://commons.wikimedia.org/wiki/File:Hanoi_Old_Quarter_(40267872341).jpg' },
  },
  {
    name: 'Queenstown',
    country: 'New Zealand',
    slug: 'queenstown',
    blurb: 'Alpine lakes and adventure trails',
    credit: { author: 'Bernard Spragg', license: 'CC0', href: 'https://commons.wikimedia.org/wiki/File:Lake_front_Queenstown_NZ_FZ200_(14554684996).jpg' },
  },
  {
    name: 'Reykjavík',
    country: 'Iceland',
    slug: 'reykjavik',
    blurb: 'Waterfalls, lava fields, northern light',
    credit: { author: 'Diego Delso', license: 'CC BY-SA 4.0', href: 'https://commons.wikimedia.org/wiki/File:Cascada_Dynjandi,_Vestfir%C3%B0ir,_Islandia,_2014-08-14,_DD_136-138_HDR.JPG' },
  },
];

const AGENTS = [
  {
    id: 'architect',
    name: 'Itinerary Architect',
    role: 'Sequences your days',
    blurb: 'Balances pace, distance and opening hours into a day-by-day plan that actually flows on the ground.',
    Icon: Compass,
  },
  {
    id: 'local',
    name: 'Local Expert',
    role: 'Cultural detail',
    blurb: 'Unwritten customs, sensory profiles and folklore — the texture of a place that guidebooks miss.',
    Icon: Brain,
  },
  {
    id: 'weather',
    name: 'Weather Analyst',
    role: 'Plans around the sky',
    blurb: 'Reads live forecasts and seasons so beaches land on sunny days and museums on the rainy ones.',
    Icon: Globe,
  },
  {
    id: 'budget',
    name: 'Budget Optimizer',
    role: 'Real numbers',
    blurb: 'Costs in local currency — meals, transit, entries — mapped precisely to the tier you choose.',
    Icon: Wallet,
  },
  {
    id: 'transport',
    name: 'Transport Planner',
    role: 'Moves you smartly',
    blurb: 'Live transit, walking times and route optimisation between every stop on your map.',
    Icon: Navigation,
  },
  {
    id: 'research',
    name: 'Ask XPLORA',
    role: 'Answers anything',
    blurb: 'Researches the live web, geocodes the location and returns a cited answer about any place on Earth.',
    Icon: Search,
  },
];

const STEPS = [
  {
    title: 'Tell us where',
    body: 'Destination, dates, pace, interests and budget — a short guided flow, not a form to fear.',
    Icon: MapPin,
  },
  {
    title: 'The agents architect',
    body: 'Six specialists work in parallel, cross-checking weather, transit, cost and culture against each other.',
    Icon: Clock,
  },
  {
    title: 'Explore your journey',
    body: 'A day-by-day plan with costs, local insight, transport routing and weather — with PDF and calendar export.',
    Icon: Sparkles,
  },
];

/* ------------------------------------------------------------------ *
 * Small building blocks
 * ------------------------------------------------------------------ */

const EASE = [0.28, 0.11, 0.32, 1] as const;

/** Fade-and-rise once the element scrolls into view. Respects reduced motion. */
function Reveal({
  children,
  delay = 0,
  className = '',
  as: Tag = 'div',
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
  as?: 'div' | 'section' | 'li' | 'article' | 'header';
}) {
  const reduced = useReducedMotion();
  if (reduced) return <Tag className={className}>{children}</Tag>;
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.65, delay, ease: EASE }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

/** Background film with graceful poster fallback, pause control and error state. */
function HeroVideo() {
  const reduced = useReducedMotion();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [failed, setFailed] = useState(false);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(!reduced);
  const [muted, setMuted] = useState(true);

  // Autoplay only when the user has not asked for reduced motion.
  useEffect(() => {
    if (reduced) {
      setPlaying(false);
      videoRef.current?.pause();
    }
  }, [reduced]);

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      void v.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
    } else {
      v.pause();
      setPlaying(false);
    }
  };

  const toggleMute = () => {
    const v = videoRef.current;
    if (!v) return;
    v.muted = !v.muted;
    setMuted(v.muted);
  };

  return (
    <div className="absolute inset-0 overflow-hidden bg-black">
      {/* Poster paints instantly, so the hero is never an empty void */}
      <img
        src="/video/hero-poster.webp"
        alt=""
        aria-hidden="true"
        className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-1000 ${
          ready && !failed ? 'opacity-0' : 'opacity-100'
        }`}
      />

      {!failed && (
        <video
          ref={videoRef}
          className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-1000 ${
            ready ? 'opacity-100' : 'opacity-0'
          }`}
          autoPlay={!reduced}
          muted
          loop
          playsInline
          preload="metadata"
          poster="/video/hero-poster.webp"
          aria-hidden="true"
          tabIndex={-1}
          onLoadedData={() => setReady(true)}
          onError={() => setFailed(true)}
        >
          <source src="/video/hero.webm" type="video/webm" />
          <source src="/video/hero.mp4" type="video/mp4" />
        </video>
      )}

      {/* Scrims: legibility on the left, depth at the bottom */}
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[linear-gradient(100deg,rgba(0,0,0,0.94)_0%,rgba(0,0,0,0.82)_38%,rgba(0,0,0,0.42)_68%,rgba(0,0,0,0.62)_100%)]"
      />
      <div
        aria-hidden="true"
        className="absolute inset-x-0 bottom-0 h-56 bg-gradient-to-t from-black via-black/70 to-transparent"
      />

      {/* Playback controls — real controls, not decoration */}
      {!failed && (
        <div className="absolute bottom-5 right-5 z-20 flex items-center gap-2">
          <button
            type="button"
            onClick={togglePlay}
            aria-label={playing ? 'Pause background film' : 'Play background film'}
            className="flex h-9 w-9 items-center justify-center rounded-full border border-white/20 bg-black/50 text-white backdrop-blur-md transition-colors hover:bg-black/75 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-300"
          >
            {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          </button>
          <button
            type="button"
            onClick={toggleMute}
            aria-label={muted ? 'Unmute background film' : 'Mute background film'}
            className="flex h-9 w-9 items-center justify-center rounded-full border border-white/20 bg-black/50 text-white backdrop-blur-md transition-colors hover:bg-black/75 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-300"
          >
            {muted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
          </button>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Page
 * ------------------------------------------------------------------ */

export default function Landing() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const reduced = useReducedMotion();
  const [dIdx, setDIdx] = useState(0);

  useEffect(() => {
    if (reduced) return;
    const t = setInterval(() => setDIdx((i) => (i + 1) % DESTINATIONS.length), 3200);
    return () => clearInterval(t);
  }, [reduced]);

  const openDestination = (name: string) => navigate(`/app?dest=${encodeURIComponent(name)}`);
  const featured = DESTINATIONS[dIdx];

  return (
    <div className="min-h-screen bg-black text-slate-200">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-white focus:px-4 focus:py-2 focus:text-[13px] focus:font-medium focus:text-black"
      >
        Skip to content
      </a>

      {/* ===== Nav ===== */}
      <header className="sticky top-0 z-40 border-b border-white/[0.08] bg-black/70 backdrop-blur-xl">
        <nav aria-label="Main" className="mx-auto flex h-14 max-w-wide items-center justify-between px-5">
          <a href="#main" className="flex items-center gap-2.5 text-slate-100" aria-label="XPLORA home">
            <BrandMark className="h-4 w-4 text-sky-300" />
            <span className="text-[15px] font-semibold tracking-tight">XPLORA</span>
          </a>

          <div className="hidden items-center gap-8 md:flex">
            <a href="#destinations" className="text-[12.5px] text-slate-300 transition-colors hover:text-white">Destinations</a>
            <a href="#agents" className="text-[12.5px] text-slate-300 transition-colors hover:text-white">Agents</a>
            <a href="#how" className="text-[12.5px] text-slate-300 transition-colors hover:text-white">How it works</a>
          </div>

          <div className="flex items-center gap-2.5">
            {user ? (
              <button
                type="button"
                onClick={() => navigate('/dashboard')}
                className="flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[12.5px] text-slate-300 transition-colors hover:text-white"
              >
                <LayoutDashboard className="h-3.5 w-3.5" />
                My trips
              </button>
            ) : (
              <button
                type="button"
                onClick={() => navigate('/login')}
                className="rounded-full px-3 py-1.5 text-[12.5px] text-slate-300 transition-colors hover:text-white"
              >
                Sign in
              </button>
            )}
            <button
              type="button"
              onClick={() => navigate('/app')}
              className="rounded-full bg-sky-500 px-4 py-1.5 text-[12.5px] font-medium text-white transition-colors hover:bg-sky-600 active:bg-sky-700"
            >
              Open app
            </button>
          </div>
        </nav>
      </header>

      <main id="main">
        {/* ===== Hero ===== */}
        <section className="relative isolate min-h-[86svh] overflow-hidden">
          <HeroVideo />

          <div className="relative mx-auto flex min-h-[86svh] max-w-wide flex-col justify-end px-5 pb-14 pt-28 sm:pb-20">
            <motion.p
              initial={reduced ? false : { opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, ease: EASE }}
              className="eyebrow mb-5"
            >
              Intelligent travel architect
            </motion.p>

            <motion.h1
              initial={reduced ? false : { opacity: 0, y: 22 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.85, delay: 0.06, ease: EASE }}
              className="max-w-3xl text-display-xl font-semibold text-white"
            >
              Journeys built
              <br />
              <span className="text-slate-400">around you.</span>
            </motion.h1>

            <motion.p
              initial={reduced ? false : { opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.14, ease: EASE }}
              className="mt-6 max-w-lg text-[16px] leading-relaxed text-slate-300 sm:text-[18px]"
            >
              Six AI agents research, design and cost your trip in real time — then hand you a
              day-by-day plan you can actually follow.
            </motion.p>

            <motion.div
              initial={reduced ? false : { opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.22, ease: EASE }}
              className="mt-8 flex flex-wrap items-center gap-3"
            >
              <button
                type="button"
                onClick={() => navigate('/app')}
                className="btn-accent group inline-flex items-center gap-2 rounded-full px-6 py-3 text-[14px] font-medium"
              >
                Plan a trip
                <ArrowRight className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-0.5" />
              </button>
              <a
                href="#destinations"
                className="group inline-flex items-center gap-1.5 rounded-full border border-white/20 bg-white/[0.06] px-5 py-3 text-[14px] text-slate-100 backdrop-blur-md transition-colors hover:border-white/35 hover:bg-white/[0.12]"
              >
                Browse destinations
                <ChevronRight className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-0.5" />
              </a>
            </motion.div>

            {/* Live destination ticker — grounds the hero in real places */}
            <motion.div
              initial={reduced ? false : { opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 1, delay: 0.5 }}
              className="mt-12 flex items-center gap-4 border-t border-white/[0.12] pt-5"
            >
              <span className="shrink-0 whitespace-nowrap text-[10px] uppercase tracking-[0.18em] text-slate-500">Featured now</span>
              <div className="flex min-w-0 items-center gap-2.5">
                <AnimatePresence mode="wait">
                  <motion.button
                    key={featured.slug}
                    type="button"
                    onClick={() => openDestination(featured.name)}
                    initial={reduced ? false : { opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={reduced ? undefined : { opacity: 0, y: -8 }}
                    transition={{ duration: 0.4, ease: EASE }}
                    className="group flex min-w-0 items-center gap-2.5 rounded-full text-[13px] text-white"
                  >
                    <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-sky-400" />
                    <span className="truncate font-medium">{featured.name}</span>
                    <span className="hidden truncate text-slate-500 sm:inline">· {featured.blurb}</span>
                    <ArrowRight className="h-3.5 w-3.5 shrink-0 text-slate-500 transition-transform duration-300 group-hover:translate-x-0.5 group-hover:text-sky-300" />
                  </motion.button>
                </AnimatePresence>
              </div>
            </motion.div>
          </div>
        </section>

        {/* ===== Destinations ===== */}
        <section id="destinations" aria-labelledby="destinations-title" className="scroll-mt-16 border-t border-white/[0.07] px-5 py-20 sm:py-28">
          <div className="mx-auto max-w-wide">
            <Reveal className="mb-10 flex flex-wrap items-end justify-between gap-6">
              <div className="max-w-xl">
                <p className="eyebrow mb-3">Where people are going</p>
                <h2 id="destinations-title" className="text-display-lg font-semibold text-white">
                  Twelve places to start.
                </h2>
              </div>
              <p className="max-w-sm text-[14px] leading-relaxed text-slate-400">
                Tap any destination to open the planner with it pre-filled. Or type anywhere on earth.
              </p>
            </Reveal>

            <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
              {DESTINATIONS.map((d, i) => (
                <li key={d.slug}>
                  <Reveal delay={Math.min(i * 0.035, 0.25)}>
                    <button
                      type="button"
                      onClick={() => openDestination(d.name)}
                      className="group relative block w-full overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0d0d0e] text-left transition-all duration-500 hover:border-white/25 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-300"
                    >
                      <div className="aspect-[3/4] w-full overflow-hidden sm:aspect-[3/2] lg:aspect-[3/4]">
                        <img
                          src={`/images/card/${d.slug}.webp`}
                          alt={`${d.name}, ${d.country}`}
                          width={720}
                          height={900}
                          loading={i < 4 ? 'eager' : 'lazy'}
                          decoding="async"
                          className="h-full w-full object-cover transition-transform duration-[900ms] ease-[cubic-bezier(0.28,0.11,0.32,1)] group-hover:scale-[1.06]"
                        />
                      </div>
                      <div
                        aria-hidden="true"
                        className="absolute inset-0 bg-gradient-to-t from-black via-black/35 to-transparent opacity-90"
                      />
                      <div className="absolute inset-x-0 bottom-0 p-4">
                        <p className="text-[10px] uppercase tracking-[0.16em] text-slate-400">{d.country}</p>
                        <p className="mt-0.5 text-[16px] font-medium leading-tight text-white">{d.name}</p>
                        <p className="mt-1 line-clamp-2 text-[11.5px] leading-snug text-slate-400 opacity-0 transition-opacity duration-300 group-hover:opacity-100">
                          {d.blurb}
                        </p>
                      </div>
                    </button>
                  </Reveal>
                </li>
              ))}
            </ul>
          </div>
        </section>

        {/* ===== Agents ===== */}
        <section id="agents" aria-labelledby="agents-title" className="scroll-mt-16 border-t border-white/[0.07] px-5 py-20 sm:py-28">
          <div className="mx-auto max-w-wide">
            <div className="grid gap-12 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)] lg:gap-16">
              <Reveal className="lg:sticky lg:top-24 lg:self-start">
                <p className="eyebrow mb-4">The crew</p>
                <h2 id="agents-title" className="text-display-lg font-semibold text-white">
                  Six agents.
                  <br />
                  <span className="text-slate-500">One journey.</span>
                </h2>
                <p className="mt-5 max-w-md text-[15px] leading-relaxed text-slate-400">
                  Each agent owns one discipline and passes its findings to the next, so nothing about
                  your trip is guessed in isolation.
                </p>
                <button
                  type="button"
                  onClick={() => navigate('/app')}
                  className="group mt-7 inline-flex items-center gap-1.5 text-[13.5px] font-medium text-sky-300 transition-colors hover:text-sky-200"
                >
                  Try it on your own trip
                  <ArrowRight className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-0.5" />
                </button>
              </Reveal>

              <ul className="grid gap-px overflow-hidden rounded-2xl border border-white/[0.08] bg-white/[0.08] sm:grid-cols-2">
                {AGENTS.map((a, i) => (
                  <li key={a.id}>
                    <Reveal delay={Math.min(i * 0.045, 0.25)} className="h-full">
                      <div className="group flex h-full flex-col bg-[#0a0a0b] p-6 transition-colors duration-400 hover:bg-[#141416]">
                        <div className="mb-4 flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] text-sky-300 transition-colors duration-300 group-hover:border-sky-500/40 group-hover:bg-sky-500/10">
                          <a.Icon className="h-4 w-4" />
                        </div>
                        <p className="mb-1 text-[10.5px] uppercase tracking-[0.14em] text-slate-500">{a.role}</p>
                        <h3 className="mb-2 text-[15.5px] font-medium text-white">{a.name}</h3>
                        <p className="text-[13px] leading-relaxed text-slate-400">{a.blurb}</p>
                      </div>
                    </Reveal>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        {/* ===== How it works ===== */}
        <section id="how" aria-labelledby="how-title" className="scroll-mt-16 border-t border-white/[0.07] px-5 py-20 sm:py-28">
          <div className="mx-auto max-w-wide">
            <Reveal className="mb-14 max-w-2xl">
              <p className="eyebrow mb-4">How it works</p>
              <h2 id="how-title" className="text-display-lg font-semibold text-white">
                Three steps to
                <br />
                <span className="text-slate-500">somewhere new.</span>
              </h2>
            </Reveal>

            <ol className="grid gap-10 md:grid-cols-3 md:gap-8">
              {STEPS.map((s, i) => (
                <li key={s.title}>
                  <Reveal delay={i * 0.08}>
                    <div className="mb-5 flex items-center gap-3">
                      <span className="font-mono text-[11px] text-sky-300">{String(i + 1).padStart(2, '0')}</span>
                      <span className="h-px flex-1 bg-gradient-to-r from-white/15 to-transparent" />
                    </div>
                    <s.Icon className="mb-4 h-5 w-5 text-slate-300" aria-hidden="true" />
                    <h3 className="mb-2 text-[18px] font-medium text-white">{s.title}</h3>
                    <p className="text-[13.5px] leading-relaxed text-slate-400">{s.body}</p>
                  </Reveal>
                </li>
              ))}
            </ol>
          </div>
        </section>

        {/* ===== Closing CTA over imagery ===== */}
        <section aria-labelledby="cta-title" className="relative isolate overflow-hidden border-t border-white/[0.07]">
          <img
            src="/images/hero/kyoto.webp"
            alt=""
            aria-hidden="true"
            loading="lazy"
            decoding="async"
            className="absolute inset-0 h-full w-full object-cover"
          />
          <div aria-hidden="true" className="absolute inset-0 bg-black/80" />
          <div className="relative mx-auto max-w-wide px-5 py-24 text-center sm:py-32">
            <Reveal>
              <h2 id="cta-title" className="mx-auto max-w-2xl text-display-md font-semibold text-white">
                Your next trip is four days away.
              </h2>
              <p className="mx-auto mt-4 max-w-md text-[15px] leading-relaxed text-slate-300">
                Describe where you want to go. The agents handle the research, pricing and routing.
              </p>
              <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
                <button
                  type="button"
                  onClick={() => navigate('/app')}
                  className="btn-accent inline-flex items-center gap-2 rounded-full px-6 py-3 text-[14px] font-medium"
                >
                  Start planning
                  <ArrowRight className="h-4 w-4" />
                </button>
                {!user && (
                  <button
                    type="button"
                    onClick={() => navigate('/signup')}
                    className="inline-flex items-center gap-2 rounded-full border border-white/20 px-5 py-3 text-[14px] text-slate-200 transition-colors hover:border-white/40 hover:text-white"
                  >
                    <CalendarDays className="h-4 w-4" />
                    Create an account
                  </button>
                )}
              </div>
            </Reveal>
          </div>
        </section>
      </main>

      {/* ===== Footer ===== */}
      <footer className="border-t border-white/[0.07] px-5 py-10">
        <div className="mx-auto max-w-wide">
          <div className="flex flex-col items-start justify-between gap-6 sm:flex-row sm:items-center">
            <div className="flex items-center gap-2.5">
              <BrandMark className="h-3.5 w-3.5 text-sky-300" />
              <span className="text-[13px] font-semibold text-slate-100">XPLORA</span>
              <span className="text-[11px] text-slate-600">Intelligent Travel Architect</span>
            </div>
            <nav aria-label="Footer" className="flex items-center gap-6 text-[11.5px] text-slate-500">
              <a href="#destinations" className="transition-colors hover:text-slate-300">Destinations</a>
              <a href="#agents" className="transition-colors hover:text-slate-300">Agents</a>
              <a href="#how" className="transition-colors hover:text-slate-300">How it works</a>
              <button type="button" onClick={() => navigate('/app')} className="transition-colors hover:text-slate-300">
                Open app
              </button>
            </nav>
          </div>

          <details className="group mt-8 border-t border-white/[0.05] pt-5">
            <summary className="cursor-pointer list-none text-[10.5px] text-slate-600 transition-colors hover:text-slate-400">
              Photography credits
              <span className="ml-1 inline-block transition-transform group-open:rotate-90">›</span>
            </summary>
            <ul className="mt-3 grid gap-x-6 gap-y-1.5 text-[10.5px] text-slate-600 sm:grid-cols-2 lg:grid-cols-3">
              {DESTINATIONS.map((d) => (
                <li key={d.slug}>
                  <a
                    href={d.credit.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="transition-colors hover:text-slate-400"
                  >
                    {d.name} — {d.credit.author} ({d.credit.license})
                  </a>
                </li>
              ))}
            </ul>
          </details>
        </div>
      </footer>
    </div>
  );
}
