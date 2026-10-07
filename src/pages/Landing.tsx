import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  motion,
  AnimatePresence,
  useMotionValue,
  useSpring,
  useInView,
} from 'framer-motion';
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
  Zap,
} from 'lucide-react';
import BrandMark from '../components/BrandMark';

const DESTINATIONS = [
  'Kyoto', 'Patagonia', 'Marrakech', 'Reykjavík', 'Lisbon', 'Queenstown',
  'Santorini', 'Banff', 'Tokyo', 'Amalfi', 'Petra', 'Hanoi',
];

// Rotating landing-page background images (auto crossfade)
const BG_IMAGES = [
  'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSZTKpBYXQfpOcWytKrwm-To58YSJ-4CDhv9B3VuX2AeQ&s=10',
  'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRtIYC56T-9s2kKeGZeM3vDx6YoT6lu4vym9MSQBLA_Cg&s=10',
  'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTMwNrhpb7LIAQXu4mxtsfRZwLThbd4IsyKjbLgYm46wQ&s=10',
  'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQpmnlpJ-vY_19_JvvfRDPGJKbqpIyeYAyxViKMSeogMA&s=10',
  'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRv2yc6OdahQmvMHXaOkU8EPeLyqjVpCWQrJXmo8f8NIA&s=10',
];

const AGENTS = [
  {
    id: 'architect',
    name: 'Itinerary Architect',
    role: 'Sequences your days',
    blurb:
      'Balances pace, distance and opening hours into a day-by-day plan that actually flows on the ground.',
    tags: ['Day-by-day', 'Pace tuning', 'Route order'],
    Icon: Compass,
  },
  {
    id: 'local',
    name: 'Local Expert',
    role: 'Cultural DNA',
    blurb:
      'Unwritten customs, sensory profiles and folklore — the texture of a place that guidebooks never capture.',
    tags: ['Customs', 'Food lore', 'Hidden spots'],
    Icon: Brain,
  },
  {
    id: 'weather',
    name: 'Weather Analyst',
    role: 'Plans around the sky',
    blurb:
      'Reads live forecasts and seasons so beaches land on sunny days and museums on the rainy ones.',
    tags: ['Live forecast', 'Seasonal fit'],
    Icon: Globe,
  },
  {
    id: 'budget',
    name: 'Budget Optimizer',
    role: 'Real numbers',
    blurb:
      'Costs in local currency — meals, transit, entries — mapped precisely to the tier you choose.',
    tags: ['Local currency', 'Tier aware'],
    Icon: Wallet,
  },
  {
    id: 'transport',
    name: 'Transport Planner',
    role: 'Moves you smartly',
    blurb:
      'Live transit, walking times and route optimization between every single stop on the map.',
    tags: ['Live transit', 'Walk times'],
    Icon: Navigation,
  },
  {
    id: 'research',
    name: 'Ask XPLORA',
    role: 'Answers anything',
    blurb:
      'Researches the live web, geocodes the location and returns a cited answer about any place on Earth.',
    tags: ['Real-time web', 'Cited answers'],
    Icon: Search,
  },
];

const STEPS = [
  {
    title: 'Tell us where',
    body: 'Destination, dates, pace, interests, budget tier and dietary needs — every detail you give reshapes the plan.',
    Icon: MapPin,
  },
  {
    title: 'Agents architect',
    body: 'Six specialized agents collaborate in real time, cross-checking weather, transit, cost and culture against each other.',
    Icon: Clock,
  },
  {
    title: 'Explore the journey',
    body: 'A rich, interactive itinerary with costs, local insights, transport routes and weather intelligence — in one clean interface.',
    Icon: Sparkles,
  },
];

const STATS = [
  { to: 180, suffix: '+', label: 'Destinations' },
  { to: 6, suffix: '', label: 'AI Agents' },
  { to: 6, suffix: '', label: 'Continents' },
  { to: 24, suffix: '/7', label: 'Planning' },
];

/* Mouse-driven 3D tilt — a playful, tactile hero mark */
function TiltCard({ children }: { children: ReactNode }) {
  const rx = useMotionValue(0);
  const ry = useMotionValue(0);
  const srx = useSpring(rx, { stiffness: 150, damping: 15 });
  const sry = useSpring(ry, { stiffness: 150, damping: 15 });
  return (
    <motion.div
      style={{ rotateX: srx, rotateY: sry, transformStyle: 'preserve-3d' }}
      className="[perspective:900px]"
      onMouseMove={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const px = (e.clientX - r.left) / r.width - 0.5;
        const py = (e.clientY - r.top) / r.height - 0.5;
        ry.set(px * 20);
        rx.set(-py * 20);
      }}
      onMouseLeave={() => {
        rx.set(0);
        ry.set(0);
      }}
    >
      {children}
    </motion.div>
  );
}

/* Count-up that fires once when scrolled into view */
function CountUp({ to, suffix = '' }: { to: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });
  const [val, setVal] = useState(0);
  useEffect(() => {
    if (!inView) return;
    let raf = 0;
    const start = performance.now();
    const dur = 1500;
    const tick = (t: number) => {
      const p = Math.min((t - start) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(Math.round(eased * to));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [inView, to]);
  return (
    <span ref={ref}>
      {val}
      {suffix}
    </span>
  );
}

export default function Landing() {
  const navigate = useNavigate();
  const [dIdx, setDIdx] = useState(0);
  const [active, setActive] = useState(0);
  const [paused, setPaused] = useState(false);
  const [step, setStep] = useState(0);
  const [bgIdx, setBgIdx] = useState(0);

  // Rotating destination in the headline
  useEffect(() => {
    const t = setInterval(
      () => setDIdx((i) => (i + 1) % DESTINATIONS.length),
      2600,
    );
    return () => clearInterval(t);
  }, []);

  // Auto-cycle the agent spotlight until the visitor hovers it
  useEffect(() => {
    if (paused) return;
    const t = setInterval(
      () => setActive((i) => (i + 1) % AGENTS.length),
      4500,
    );
    return () => clearInterval(t);
  }, [paused]);

  // Timed background image rotation
  useEffect(() => {
    const t = setInterval(
      () => setBgIdx((i) => (i + 1) % BG_IMAGES.length),
      6000,
    );
    return () => clearInterval(t);
  }, []);

  const A = AGENTS[active];

  return (
    <div className="main-gradient text-on-image min-h-screen text-slate-200 overflow-hidden">
      {/* ===== ROTATING BACKGROUND ===== */}
      <div className="fixed inset-0 z-0" aria-hidden="true">
        {BG_IMAGES.map((src, i) => (
          <div
            key={src}
            className="absolute inset-0 transition-all duration-[2000ms] ease-in-out"
            style={{
              backgroundImage: `url("${src}")`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
              opacity: i === bgIdx ? 1 : 0,
              transform: i === bgIdx ? 'scale(1.08)' : 'scale(1)',
            }}
          />
        ))}
      </div>

      {/* ===== HERO ===== */}
      <section className="relative z-10 min-h-screen flex flex-col items-center justify-center px-6 text-center">
        <div className="absolute inset-x-0 top-0 h-[1px] bg-gradient-to-r from-transparent via-sky-400/30 to-transparent" />

        {/* Logo mark with interactive tilt */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="mb-10"
        >
          <TiltCard>
            <div className="relative">
              <div className="absolute inset-0 bg-sky-400 blur-[100px] opacity-20 animate-pulse-soft" />
              <motion.div
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }}
                className="bg-gradient-to-br from-sky-400/15 via-sky-400/5 to-sky-400/10 p-10 rounded-[2.5rem] border border-sky-400/20 shadow-2xl relative z-10 backdrop-blur-xl"
              >
                <BrandMark className="w-20 h-20 text-sky-300 relative z-10 drop-shadow-[0_0_20px_rgba(56,189,248,0.4)]" />
              </motion.div>
            </div>
          </TiltCard>
        </motion.div>

        {/* Wordmark */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
          className="mb-4"
        >
          <h1 className="text-7xl md:text-8xl lg:text-9xl font-bold tracking-tight text-white leading-none">
            <span className="wordmark">XPLORA</span>
          </h1>
        </motion.div>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.25, ease: [0.16, 1, 0.3, 1] }}
          className="text-sm md:text-base text-slate-200 uppercase tracking-[0.35em] font-medium mb-8"
        >
          Intelligent Travel Architect
        </motion.p>

        {/* Headline with rotating destination */}
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.35, ease: [0.16, 1, 0.3, 1] }}
          className="text-4xl md:text-6xl lg:text-7xl font-bold text-white mb-6 leading-tight max-w-4xl"
        >
          Plan a trip to{' '}
          <span className="relative inline-block">
            <AnimatePresence mode="wait">
              <motion.span
                key={DESTINATIONS[dIdx]}
                initial={{ y: '0.35em', opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: '-0.35em', opacity: 0 }}
                transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
                className="accent-label italic inline-block"
              >
                {DESTINATIONS[dIdx]}
              </motion.span>
            </AnimatePresence>
          </span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.45, ease: [0.16, 1, 0.3, 1] }}
          className="text-base md:text-lg text-slate-100 max-w-2xl mb-12 leading-relaxed font-light"
        >
          Six AI agents research, design and cost your trip in real time — then
          hand you a day-by-day plan you can actually follow.
        </motion.p>

        {/* The only "VIEW THE APPLICATION" call-to-action */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.55, ease: [0.16, 1, 0.3, 1] }}
          className="flex flex-col sm:flex-row items-center gap-4 mb-20"
        >
          <button
            onClick={() => navigate('/app')}
            className="btn-accent group relative font-bold py-4 px-10 rounded-2xl flex items-center gap-3 tracking-[0.12em] text-sm overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700 -skew-x-12 translate-x-[-100%] group-hover:translate-x-[100%] duration-1000" />
            <Zap className="w-5 h-5 relative z-10" />
            <span className="relative z-10">VIEW THE APPLICATION</span>
            <ArrowRight className="w-5 h-5 relative z-10 group-hover:translate-x-1 transition-transform duration-300" />
          </button>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 1 }}
          className="absolute bottom-10 left-1/2 -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
            className="w-6 h-10 rounded-full border-2 border-white/40 flex items-start justify-center p-1.5"
          >
            <motion.div
              animate={{ opacity: [0.3, 1, 0.3], height: ['4px', '8px', '4px'] }}
              transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
              className="w-1 bg-sky-400 rounded-full"
            />
          </motion.div>
        </motion.div>
      </section>

      {/* ===== DESTINATION MARQUEE ===== */}
      <div className="marquee-mask relative z-10 overflow-hidden border-y border-white/5 py-5" aria-hidden="true">
        <div className="animate-marquee flex w-max">
          {[...DESTINATIONS, ...DESTINATIONS].map((d, i) => (
            <span
              key={i}
              className="mr-10 inline-flex items-center gap-10 text-slate-300 text-sm uppercase tracking-[0.3em]"
            >
              {d}
              <span className="w-1 h-1 rounded-full bg-sky-400/50" />
            </span>
          ))}
        </div>
      </div>

      {/* ===== STATS ===== */}
      <section className="relative z-10 px-6 py-16">
        <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">
          {STATS.map((s) => (
            <div key={s.label} className="text-center">
              <p className="text-4xl md:text-5xl font-bold text-white">
                <CountUp to={s.to} suffix={s.suffix} />
              </p>
              <p className="text-[11px] uppercase tracking-[0.25em] text-slate-300 mt-2">
                {s.label}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* ===== AGENT SPOTLIGHT (interactive) ===== */}
      <section className="relative z-10 px-6 py-24 md:py-32">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="text-center mb-14"
          >
            <p className="text-[10px] font-bold text-sky-300 uppercase tracking-[0.3em] mb-4">
              Meet the crew
            </p>
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Six agents.{' '}
              <span className="accent-label italic">One journey.</span>
            </h2>
            <p className="text-slate-200 max-w-xl mx-auto text-base font-light">
              Tap an agent to see what it handles — or let it cycle through.
            </p>
          </motion.div>

          <div
            onMouseEnter={() => setPaused(true)}
            onMouseLeave={() => setPaused(false)}
          >
            <div className="flex flex-wrap justify-center gap-2 mb-10">
              {AGENTS.map((a, i) => (
                <button
                  key={a.id}
                  onClick={() => setActive(i)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm border backdrop-blur-md transition-all duration-300 ${
                    i === active
                      ? 'border-sky-400 bg-sky-400 text-[#05070a] font-semibold shadow-[0_6px_20px_rgba(56,189,248,0.35)]'
                      : 'border-white/15 bg-[#0c0e12]/80 text-slate-200 hover:bg-[#0c0e12]/95 hover:text-white'
                  }`}
                >
                  <a.Icon className="w-4 h-4" />
                  {a.name}
                </button>
              ))}
            </div>

            <AnimatePresence mode="wait">
              <motion.div
                key={A.id}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -16 }}
                transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                className="flat-card p-8 md:p-10 max-w-3xl mx-auto text-left"
              >
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-12 h-12 rounded-xl border border-sky-400/20 bg-sky-400/10 flex items-center justify-center text-sky-300">
                    <A.Icon className="w-6 h-6" />
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-[0.3em] text-sky-300">
                      {A.role}
                    </p>
                    <h3 className="text-xl font-bold text-white">{A.name}</h3>
                  </div>
                </div>
                <p className="text-slate-400 font-light leading-relaxed mb-5">
                  {A.blurb}
                </p>
                <div className="flex flex-wrap gap-2">
                  {A.tags.map((t) => (
                    <span
                      key={t}
                      className="text-xs px-3 py-1 rounded-full border border-white/15 bg-white/[0.06] text-slate-200"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </section>

      {/* ===== HOW IT WORKS (interactive stepper) ===== */}
      <section className="relative z-10 px-6 py-24 md:py-32">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="text-center mb-14"
          >
            <p className="text-[10px] font-bold text-sky-300 uppercase tracking-[0.3em] mb-4">
              How it works
            </p>
            <h2 className="text-4xl md:text-5xl font-bold text-white">
              Three steps to{' '}
              <span className="accent-label italic">extraordinary</span>
            </h2>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            {STEPS.map((s, i) => (
              <button
                key={s.title}
                onClick={() => setStep(i)}
                className={`text-left p-5 rounded-2xl border backdrop-blur-md transition-all duration-300 ${
                  i === step
                    ? 'border-sky-400/60 bg-[#0c0e12]/90'
                    : 'border-white/15 bg-[#0c0e12]/75 hover:bg-[#0c0e12]/90'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <span
                    className={`w-9 h-9 rounded-lg flex items-center justify-center border transition-colors duration-300 ${
                      i === step
                        ? 'border-sky-400/50 text-sky-300'
                        : 'border-white/15 text-slate-300'
                    }`}
                  >
                    <s.Icon className="w-4 h-4" />
                  </span>
                  <span className="text-[10px] text-slate-400 tracking-[0.3em]">
                    0{i + 1}
                  </span>
                </div>
                <p className="font-bold text-white">{s.title}</p>
              </button>
            ))}
          </div>

          <div className="h-[2px] w-full bg-white/5 rounded-full overflow-hidden mb-8">
            <motion.div
              className="h-full bg-sky-400/60"
              animate={{ width: `${((step + 1) / STEPS.length) * 100}%` }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            />
          </div>

          <AnimatePresence mode="wait">
            <motion.p
              key={step}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.35 }}
              className="text-slate-100 font-light max-w-xl mx-auto text-center leading-relaxed"
            >
              {STEPS[step].body}
            </motion.p>
          </AnimatePresence>
        </div>
      </section>

      {/* ===== FOOTER ===== */}
      <footer className="relative z-10 border-t border-white/5 py-12 px-6">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-sky-400/20 to-sky-400/5 p-2 rounded-xl border border-sky-400/10">
              <BrandMark className="w-4 h-4 text-sky-300" />
            </div>
            <span className="text-sm font-bold text-white tracking-tight">
              <span className="wordmark">XPLORA</span>
            </span>
            <span className="text-[10px] text-slate-600 italic tracking-wider">
              Intelligent Travel Architect
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
