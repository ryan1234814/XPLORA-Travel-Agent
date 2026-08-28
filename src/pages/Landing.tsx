import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Diamond,
  MapPin,
  Search,
  Compass,
  Globe,
  Sparkles,
  ArrowRight,
  Zap,
  Shield,
  Clock,
  Brain,
  ChevronRight,
} from 'lucide-react';

const features = [
  {
    icon: <Compass className="w-6 h-6" />,
    title: 'AI Itinerary Architect',
    description:
      'Multi-agent system designs bespoke day-by-day itineraries tailored to your pace, interests, and budget — from Essential to Legendary.',
    gradient: 'from-primary/20 to-primary/5',
    border: 'border-primary/20',
    iconColor: 'text-primary',
  },
  {
    icon: <Search className="w-6 h-6" />,
    title: 'Ask XPLORA',
    description:
      'Ask anything about any place on Earth. XPLORA researches the web in real-time, geocodes the location, and delivers cited answers.',
    gradient: 'from-teal/20 to-teal/5',
    border: 'border-teal/20',
    iconColor: 'text-teal',
  },
  {
    icon: <Brain className="w-6 h-6" />,
    title: 'Local Expert Intelligence',
    description:
      'Unwritten customs, sensory profiles, folklore heritage — deep cultural DNA that guidebooks can never capture.',
    gradient: 'from-amber/20 to-amber/5',
    border: 'border-amber/20',
    iconColor: 'text-amber',
  },
  {
    icon: <Globe className="w-6 h-6" />,
    title: 'Live Transport & Weather',
    description:
      'Real-time transit data, route optimization, and weather analysis so you move smarter and pack right.',
    gradient: 'from-rose/20 to-rose/5',
    border: 'border-rose/20',
    iconColor: 'text-rose',
  },
  {
    icon: <Shield className="w-6 h-6" />,
    title: 'Budget Intelligence',
    description:
      'Precision cost breakdowns in local currency — meals, transport, entries — mapped to your chosen budget tier.',
    gradient: 'from-indigo/20 to-indigo/5',
    border: 'border-indigo/20',
    iconColor: 'text-indigo',
  },
  {
    icon: <Sparkles className="w-6 h-6" />,
    title: '180+ Destinations',
    description:
      'From Kyoto temples to Patagonian glaciers — XPLORA has deep knowledge across six continents and counting.',
    gradient: 'from-fuchsia/20 to-fuchsia/5',
    border: 'border-fuchsia/20',
    iconColor: 'text-fuchsia',
  },
];

const STAR_COLORS = ['#38bdf8', '#2dd4bf', '#fbbf24', '#fb7185', '#e879f9', '#ffffff'];

// Seeded pseudo-random for stable star positions
function seededRandom(seed: number): number {
  const x = Math.sin(seed * 9301 + 49297) * 49297;
  return x - Math.floor(x);
}

const starData = Array.from({ length: 60 }, (_, i) => ({
  left: `${seededRandom(i * 7 + 1) * 100}%`,
  top: `${seededRandom(i * 13 + 3) * 100}%`,
  width: `${seededRandom(i * 3 + 5) * 2.5 + 1}px`,
  height: `${seededRandom(i * 11 + 7) * 2.5 + 1}px`,
  animationDelay: `${seededRandom(i * 17 + 9) * 15}s`,
  animationDuration: `${seededRandom(i * 19 + 11) * 10 + 10}s`,
  background: STAR_COLORS[Math.floor(seededRandom(i * 23 + 13) * 6)],
  opacity: seededRandom(i * 29 + 15) * 0.5 + 0.1,
}));

export default function Landing() {
  const navigate = useNavigate();
  const stars = useMemo(() => starData, []);

  return (
    <div className="main-gradient min-h-screen font-outfit text-slate-200 overflow-hidden">
      {/* Floating particles */}
      <div className="stars-container" aria-hidden="true">
        {stars.map((s, i) => (
          <div key={i} className="star" style={s} />
        ))}
      </div>

      {/* Ambient glow orbs */}
      <div className="glow-orb glow-orb--primary" style={{ top: '5%', left: '-8%' }} />
      <div className="glow-orb glow-orb--accent" style={{ top: '50%', right: '-5%' }} />
      <div className="glow-orb glow-orb--amber" style={{ bottom: '5%', left: '30%' }} />
      <div className="glow-orb glow-orb--rose" style={{ top: '70%', left: '-10%' }} />

      {/* Ambient glow overlays */}
      <div className="ambient-glow" />
      <div className="ambient-glow--bottom" />

      {/* ===== HERO SECTION ===== */}
      <section className="relative z-10 min-h-screen flex flex-col items-center justify-center px-6 text-center">
        <div className="absolute inset-x-0 top-0 h-[1px] bg-gradient-to-r from-transparent via-primary/30 to-transparent" />

        {/* Logo mark */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="mb-10"
        >
          <div className="relative">
            <div className="absolute inset-0 bg-primary blur-[100px] opacity-20 animate-pulse-soft" />
            <motion.div
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }}
              className="bg-gradient-to-br from-primary/15 via-primary/5 to-secondary/10 p-10 rounded-[2.5rem] border border-primary/20 shadow-2xl relative z-10 backdrop-blur-xl"
            >
              <div className="absolute inset-0 rounded-[2.5rem] bg-gradient-to-br from-primary/10 to-transparent opacity-50" />
              <Diamond className="w-20 h-20 text-primary relative z-10 drop-shadow-[0_0_20px_rgba(56,189,248,0.4)]" />
            </motion.div>
          </div>
        </motion.div>

        {/* Title */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
          className="mb-4"
        >
          <h1 className="text-7xl md:text-8xl lg:text-9xl font-bold tracking-tight text-white leading-none">
            <span className="text-gradient-shimmer">XPLORA</span>
          </h1>
        </motion.div>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.25, ease: [0.16, 1, 0.3, 1] }}
          className="text-sm md:text-base text-slate-500 uppercase tracking-[0.35em] font-medium mb-8"
        >
          Intelligent Travel Architect
        </motion.p>

        {/* Headline */}
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.35, ease: [0.16, 1, 0.3, 1] }}
          className="text-3xl md:text-5xl lg:text-6xl font-bold text-white mb-6 leading-tight max-w-4xl"
        >
          Craft Your{' '}
          <span className="text-gradient-rainbow italic">Bespoke</span> Travel
          <br />
          Narrative
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.45, ease: [0.16, 1, 0.3, 1] }}
          className="text-base md:text-lg text-slate-400 max-w-2xl mb-12 leading-relaxed font-light"
        >
          XPLORA transcends standard planning. A constellation of AI agents
          researches, designs, and curates intelligent travel experiences that
          resonate with your soul.
        </motion.p>

        {/* CTA Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.55, ease: [0.16, 1, 0.3, 1] }}
          className="flex flex-col sm:flex-row items-center gap-4 mb-20"
        >
          <button
            onClick={() => navigate('/app')}
            className="group relative text-white font-bold py-4 px-10 rounded-2xl shadow-[0_5px_30px_rgba(56,189,248,0.35)] hover:shadow-[0_8px_50px_rgba(56,189,248,0.55)] hover:-translate-y-0.5 active:translate-y-0 transition-all duration-300 flex items-center gap-3 tracking-[0.12em] text-sm overflow-hidden"
            style={{
              background:
                'linear-gradient(135deg, #38bdf8 0%, #0284c7 25%, #2dd4bf 65%, #0d9488 100%)',
              backgroundSize: '200% 200%',
              animation: 'gradient-shift 4s ease-in-out infinite',
            }}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/15 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700 -skew-x-12 translate-x-[-100%] group-hover:translate-x-[100%] duration-1000" />
            <Zap className="w-5 h-5 relative z-10" />
            <span className="relative z-10">EXPLORE THE APP</span>
            <ArrowRight className="w-5 h-5 relative z-10 group-hover:translate-x-1 transition-transform duration-300" />
          </button>

          <a
            href="https://github.com/ryan1234814/XPLORA-Travel-Agent"
            target="_blank"
            rel="noopener noreferrer"
            className="text-slate-400 font-medium py-4 px-8 rounded-2xl border border-white/10 hover:border-white/20 hover:text-white hover:bg-white/[0.04] transition-all duration-300 text-sm flex items-center gap-2"
          >
            View on GitHub
            <ChevronRight className="w-4 h-4" />
          </a>
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
            className="w-6 h-10 rounded-full border-2 border-white/15 flex items-start justify-center p-1.5"
          >
            <motion.div
              animate={{ opacity: [0.3, 1, 0.3], height: ['4px', '8px', '4px'] }}
              transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
              className="w-1 bg-primary rounded-full"
            />
          </motion.div>
        </motion.div>
      </section>

      {/* ===== FEATURES SECTION ===== */}
      <section className="relative z-10 px-6 py-24 md:py-32">
        <div className="max-w-6xl mx-auto">
          {/* Section header */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="text-center mb-20"
          >
            <p className="text-[10px] font-bold text-primary uppercase tracking-[0.3em] mb-4">
              Powered by AI
            </p>
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Six Agents.{' '}
              <span className="text-gradient-rainbow italic">One Vision.</span>
            </h2>
            <p className="text-slate-400 max-w-xl mx-auto text-base font-light">
              A coordinated team of specialized AI agents working together to
              craft your perfect journey.
            </p>
          </motion.div>

          {/* Feature grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, i) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-50px' }}
                transition={{
                  duration: 0.6,
                  delay: i * 0.1,
                  ease: [0.16, 1, 0.3, 1],
                }}
                className={`glass-card-premium p-7 group hover:shadow-[0_12px_48px_rgba(56,189,248,0.08)] transition-all duration-300`}
              >
                <div
                  className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.gradient} border ${feature.border} flex items-center justify-center mb-5 ${feature.iconColor} group-hover:scale-110 transition-transform duration-300`}
                >
                  {feature.icon}
                </div>
                <h3 className="text-lg font-bold text-white mb-2.5">
                  {feature.title}
                </h3>
                <p className="text-sm text-slate-400 leading-relaxed font-light">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== HOW IT WORKS ===== */}
      <section className="relative z-10 px-6 py-24 md:py-32">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="text-center mb-20"
          >
            <p className="text-[10px] font-bold text-teal uppercase tracking-[0.3em] mb-4">
              How It Works
            </p>
            <h2 className="text-4xl md:text-5xl font-bold text-white">
              Three Steps to{' '}
              <span className="text-gradient-teal italic">Extraordinary</span>
            </h2>
          </motion.div>

          <div className="space-y-16">
            {[
              {
                step: '01',
                title: 'Tell Us Where',
                description:
                  'Enter your destination, set your dates, pick your pace and interests. From budget tier to dietary needs — every detail matters.',
                icon: <MapPin className="w-6 h-6 text-primary" />,
                color: 'primary',
              },
              {
                step: '02',
                title: 'AI Agents Architect',
                description:
                  'Six specialized agents — Travel Advisor, Weather Analyst, Budget Optimizer, Local Expert, Transport Planner, and Itinerary Architect — collaborate in real-time.',
                icon: <Clock className="w-6 h-6 text-teal" />,
                color: 'teal',
              },
              {
                step: '03',
                title: 'Explore Your Journey',
                description:
                  'Receive a rich, interactive itinerary with cost breakdowns, local cultural insights, transport routes, and weather intelligence — all in one beautiful interface.',
                icon: <Sparkles className="w-6 h-6 text-amber" />,
                color: 'amber',
              },
            ].map((item, i) => (
              <motion.div
                key={item.step}
                initial={{ opacity: 0, x: i % 2 === 0 ? -30 : 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: '-80px' }}
                transition={{
                  duration: 0.7,
                  delay: i * 0.15,
                  ease: [0.16, 1, 0.3, 1],
                }}
                className="flex items-start gap-8 md:gap-12"
              >
                <div
                  className={`shrink-0 w-16 h-16 rounded-2xl bg-${item.color}/10 border border-${item.color}/20 flex items-center justify-center relative`}
                >
                  {item.icon}
                  <span className="absolute -top-2 -right-2 text-[9px] font-bold text-slate-500 bg-[#0c0e12] px-1.5 py-0.5 rounded-md border border-white/5">
                    {item.step}
                  </span>
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white mb-2">
                    {item.title}
                  </h3>
                  <p className="text-sm text-slate-400 leading-relaxed font-light max-w-lg">
                    {item.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== CTA SECTION ===== */}
      <section className="relative z-10 px-6 py-24 md:py-32">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="max-w-4xl mx-auto text-center"
        >
          <div className="glass-card-premium p-12 md:p-16 relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-teal/5" />
            <div className="relative z-10">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-teal/10 border border-primary/20 flex items-center justify-center mx-auto mb-8">
                <Diamond className="w-8 h-8 text-primary drop-shadow-[0_0_12px_rgba(56,189,248,0.4)]" />
              </div>
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-4 leading-tight">
                Ready to Discover
                <br />
                <span className="text-gradient-rainbow italic">
                  Something Extraordinary?
                </span>
              </h2>
              <p className="text-slate-400 mb-10 max-w-lg mx-auto font-light">
                Let XPLORA's AI agents research, design, and craft a travel
                experience uniquely yours.
              </p>
              <button
                onClick={() => navigate('/app')}
                className="group relative text-white font-bold py-4 px-12 rounded-2xl shadow-[0_5px_30px_rgba(56,189,248,0.35)] hover:shadow-[0_8px_50px_rgba(56,189,248,0.55)] hover:-translate-y-0.5 active:translate-y-0 transition-all duration-300 flex items-center gap-3 tracking-[0.12em] text-sm mx-auto overflow-hidden"
                style={{
                  background:
                    'linear-gradient(135deg, #38bdf8 0%, #0284c7 25%, #2dd4bf 65%, #0d9488 100%)',
                  backgroundSize: '200% 200%',
                  animation: 'gradient-shift 4s ease-in-out infinite',
                }}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/15 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700 -skew-x-12 translate-x-[-100%] group-hover:translate-x-[100%] duration-1000" />
                <Zap className="w-5 h-5 relative z-10" />
                <span className="relative z-10">EXPLORE THE APP</span>
                <ArrowRight className="w-5 h-5 relative z-10 group-hover:translate-x-1 transition-transform duration-300" />
              </button>
            </div>
          </div>
        </motion.div>
      </section>

      {/* ===== FOOTER ===== */}
      <footer className="relative z-10 border-t border-white/5 py-12 px-6">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-primary/20 to-primary/5 p-2 rounded-xl border border-primary/10">
              <Diamond className="w-4 h-4 text-primary" />
            </div>
            <span className="text-sm font-bold text-white tracking-tight">
              <span className="text-gradient-shimmer">XPLORA</span>
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
