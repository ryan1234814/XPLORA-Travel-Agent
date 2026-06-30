import { useState } from 'react';
import {
  Diamond,
  MapPin,
  Calendar,
  Wallet,
  Heart,
  ChevronRight,
  Search,
  RefreshCcw,
  ArrowRight,
  Sun,
  Camera,
  Utensils,
  History,
  Activity,
  Trees as Tree,
  Plane,
  Train,
  Car,
  Bus,
  Clock,
  Navigation,
  Zap,
  Award,
  ShieldCheck,
  Info,
  X
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000');

// Types
interface ActivityData {
  time: string;
  title: string;
  description: string;
  location: string;
  tag: string;
  map_query?: string;
  transport_to_next?: {
    mode: string;
    duration: string;
    cost: string;
    instructions: string;
  };
}

interface DayData {
  day_number: number;
  day_name: string;
  theme: string;
  activities: ActivityData[];
}

interface ItineraryData {
  trip_title: string;
  overview: string;
  sustainability_score: number;
  price_range: string;
  concierge_note: string;
  days: DayData[];
}

interface MobilityData {
  flights?: any;
  regional_trains_buses?: any;
  car_rentals?: any;
  airport_transfers?: any;
  local_transport?: any;
  route_optimization?: any;
}

interface WeatherData {
  destination?: string;
  temperature_c?: {
    expected_low?: number;
    expected_high?: number;
    typical_range?: string;
    notes?: string;
  };
  conditions_summary?: string;
}

const interestsOptions = [
  { id: 'Wellness', icon: <Heart className="w-4 h-4" /> },
  { id: 'Gastronomy', icon: <Utensils className="w-4 h-4" /> },
  { id: 'Photography', icon: <Camera className="w-4 h-4" /> },
  { id: 'History', icon: <History className="w-4 h-4" /> },
  { id: 'Adventure', icon: <Activity className="w-4 h-4" /> },
  { id: 'Art', icon: <Sun className="w-4 h-4" /> }
];

const budgetTiers = ["Essential", "Premier", "Elite", "Legendary"];

function App() {
  const [destination, setDestination] = useState('');
  const [origin, setOrigin] = useState('');
  const [duration, setDuration] = useState(3);
  const [budget, setBudget] = useState('Premier');
  const [selectedInterests, setSelectedInterests] = useState(['Wellness', 'Gastronomy']);
  const [isLoading, setIsLoading] = useState(false);
  const [itinerary, setItinerary] = useState<ItineraryData | null>(null);
  const [mobility, setMobility] = useState<MobilityData | null>(null);
  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [localExpert, setLocalExpert] = useState<any>(null);
  const [isIntelligenceOpen, setIsIntelligenceOpen] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [expandedMobility, setExpandedMobility] = useState<string | null>(null);

  const toggleInterest = (id: string) => {
    setSelectedInterests(prev =>
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
  };

  const handleGenerate = async () => {
    if (!destination) {
      setError("Please define a destination.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setItinerary(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/generate-itinerary`, {
        origin,
        destination,
        duration,
        budget,
        interests: selectedInterests
      });

      const data = response.data;
      console.log('[DEBUG] Full API response keys:', Object.keys(data));

      const parseField = (field: any) => {
        if (!field) return null;
        let obj = field.output || field;
        if (typeof obj === 'string') {
          try {
            // Find JSON in string if it's there
            const match = obj.match(/\{[\s\S]*\}/);
            if (match) return JSON.parse(match[0]);
            return obj;
          } catch (e) { return obj; }
        }
        return obj;
      };

      // Parse weather with extra robustness
      const rawWeather = data.weather_analyst;
      console.log('[DEBUG] Raw weather_analyst:', JSON.stringify(rawWeather).substring(0, 300));
      let parsedWeather = parseField(rawWeather);
      // If parseField returned a string (e.g. from response field), try to parse JSON from it
      if (typeof parsedWeather === 'string') {
        try {
          const match = parsedWeather.match(/\{[\s\S]*\}/);
          if (match) parsedWeather = JSON.parse(match[0]);
        } catch (e) { /* keep as string */ }
      }
      // Fallback: if output didn't have temperature_c, try parsing the response field
      if (parsedWeather && typeof parsedWeather === 'object' && !parsedWeather.temperature_c && rawWeather?.response) {
        try {
          const fromResponse = JSON.parse(rawWeather.response);
          if (fromResponse.temperature_c) {
            parsedWeather = fromResponse;
          }
        } catch (e) { /* ignore */ }
      }
      console.log('[DEBUG] Parsed weather:', JSON.stringify(parsedWeather).substring(0, 300));

      setItinerary(parseField(data.itinerary_planner));
      setMobility(parseField(data.transport_mobility));
      setWeather(parsedWeather);
      setLocalExpert(parseField(data.local_expert));

      setActiveTab(0);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred during generation. Our servers are currently at capacity.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setItinerary(null);
    setMobility(null);
    setWeather(null);
    setLocalExpert(null);
    setDestination('');
    setOrigin('');
    setDuration(3);
    setBudget('Premier');
    setSelectedInterests(['Wellness', 'Gastronomy']);
    setError(null);
    setExpandedMobility(null);
  };

  return (
    <div className="main-gradient min-h-screen font-outfit text-slate-200">
      <div className="flex h-screen overflow-hidden">
        {/* Sidebar */}
        <aside className="w-80 bg-[#0c0e12] border-r border-white/5 flex flex-col shrink-0 z-20">
          <div className="p-8 pb-4 flex items-center gap-4 relative">
            <div className="absolute inset-x-6 -bottom-2 h-[1px] bg-gradient-to-r from-transparent via-primary/30 to-transparent"></div>
            <div className="bg-gradient-to-br from-primary/25 to-secondary/20 p-3 rounded-2xl border border-primary/10 shadow-[0_0_30px_rgba(164,140,244,0.15)]">
              <Diamond className="w-7 h-7 text-primary drop-shadow-[0_0_8px_rgba(164,140,244,0.5)]" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white">VELURA</h1>
              <p className="text-[10px] text-slate-500 italic tracking-[0.15em] mt-0.5">Personal Travel Architect</p>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Origin (Optional)</label>
              <div className="relative group input-glow rounded-xl">
                <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-primary transition-colors duration-300" />
                <input
                  type="text"
                  value={origin}
                  onChange={(e) => setOrigin(e.target.value)}
                  placeholder="e.g. New Delhi (DEL)"
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pl-10 pr-4 text-sm focus:border-primary/50 focus:bg-primary/[0.06] focus:shadow-[0_0_20px_rgba(164,140,244,0.06)] transition-all duration-300 outline-none"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Destination</label>
              <div className="relative group input-glow rounded-xl">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-primary transition-colors duration-300" />
                <input
                  type="text"
                  value={destination}
                  onChange={(e) => setDestination(e.target.value)}
                  placeholder="e.g. Kyoto, Japan"
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pl-10 pr-4 text-sm focus:border-primary/50 focus:bg-primary/[0.06] focus:shadow-[0_0_20px_rgba(164,140,244,0.06)] transition-all duration-300 outline-none font-medium"
                />
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center px-1">
                <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em]">Duration</label>
                <span className="text-xs font-bold text-primary px-3 py-1.5 bg-gradient-to-r from-primary/15 to-primary/5 rounded-lg border border-primary/10 shadow-[0_0_15px_rgba(164,140,244,0.08)]">{duration} Days</span>
              </div>
              <input
                type="range"
                min="1"
                max="14"
                value={duration}
                onChange={(e) => setDuration(parseInt(e.target.value))}
                className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Tier</label>
              <select
                value={budget}
                onChange={(e) => setBudget(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 px-4 text-sm focus:border-primary/50 transition-all duration-300 outline-none appearance-none cursor-pointer hover:bg-white/[0.07]"
              >
                {budgetTiers.map(tier => (
                  <option key={tier} value={tier} className="bg-[#0c0e12]">{tier}</option>
                ))}
              </select>
            </div>

            <div className="space-y-3">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Focus</label>
              <div className="grid grid-cols-2 gap-2">
                {interestsOptions.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => toggleInterest(item.id)}
                    className={`flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all duration-300 border ${selectedInterests.includes(item.id)
                      ? 'bg-gradient-to-br from-primary/25 to-primary/10 border-primary/50 text-white shadow-[0_0_20px_rgba(164,140,244,0.1)] hover:shadow-[0_0_30px_rgba(164,140,244,0.2)]'
                      : 'bg-white/5 border-white/5 text-slate-400 hover:bg-white/10 hover:border-white/10 hover:text-slate-300'
                      }`}
                  >
                    <span className={`transition-all duration-300 ${selectedInterests.includes(item.id) ? 'text-primary scale-110' : 'text-slate-500'}`}>
                      {item.icon}
                    </span>
                    {item.id}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="p-6 pt-2 space-y-3 border-t border-white/5 mt-auto">
            <button
              onClick={handleGenerate}
              disabled={isLoading}
              className="w-full bg-gradient-to-br from-[#a48cf4] via-[#8b72e0] to-[#6e56cf] text-white font-bold py-4 px-6 rounded-xl shadow-[0_5px_25px_rgba(164,140,244,0.3)] hover:shadow-[0_8px_40px_rgba(164,140,244,0.5)] hover:-translate-y-0.5 active:translate-y-0 transition-all duration-300 disabled:opacity-50 disabled:translate-y-0 flex items-center justify-center gap-2.5 tracking-[0.12em] text-xs relative overflow-hidden group/btn"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent opacity-0 group-hover/btn:opacity-100 transition-opacity duration-700 -skew-x-12 translate-x-[-100%] group-hover/btn:translate-x-[100%] duration-1000"></div>
              {isLoading ? (
                <>
                  <RefreshCcw className="w-4 h-4 animate-spin text-white/70" />
                  DESIGNING YOUR VOYAGE...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  DESIGN ITINERARY
                </>
              )}
            </button>
            <button
              onClick={handleReset}
              className="w-full bg-white/[0.04] text-slate-400 font-bold py-3.5 px-6 rounded-xl hover:bg-white/[0.08] hover:text-slate-300 transition-all duration-300 text-xs uppercase tracking-widest border border-white/[0.06] hover:border-white/10"
            >
              RESET
            </button>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 overflow-y-auto scroll-smooth relative">
          <AnimatePresence mode="wait">
            {!itinerary && !isLoading ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 1.05 }}
                key="welcome"
                className="h-full flex flex-col items-center justify-center p-8 text-center max-w-3xl mx-auto relative"
              >
                <div className="ambient-glow"></div>
                <div className="ambient-glow--bottom"></div>
                
                <div className="relative mb-14">
                  <div className="absolute inset-0 bg-primary blur-[120px] opacity-25 animate-pulse-soft"></div>
                  <motion.div
                    animate={{ y: [0, -12, 0] }}
                    transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
                    className="bg-gradient-to-br from-primary/15 via-primary/5 to-secondary/10 p-12 rounded-[3rem] border border-primary/20 shadow-2xl relative z-10 backdrop-blur-xl"
                  >
                    <div className="absolute inset-0 rounded-[3rem] bg-gradient-to-br from-primary/10 to-transparent opacity-50"></div>
                    <Diamond className="w-24 h-24 text-primary relative z-10 drop-shadow-[0_0_20px_rgba(164,140,244,0.4)]" />
                  </motion.div>
                </div>
                
                <h2 className="text-6xl md:text-7xl font-bold mb-6 text-white leading-tight tracking-tight">
                  Craft Your{' '}
                  <span className="bg-gradient-to-r from-primary via-accent to-accent2 bg-clip-text text-transparent italic">Bespoke</span>
                  {' '}Narrative
                </h2>
                <p className="text-lg md:text-xl text-slate-400 mb-14 leading-relaxed font-light max-w-2xl">
                  Velura transcends standard planning. We curate high-end travel experiences
                  that resonate with your soul and define your legacy.
                </p>
                
                <div className="grid grid-cols-3 gap-8 w-full max-w-xl mx-auto bg-white/[0.04] p-8 rounded-3xl border border-white/[0.06] stagger-fade-in">
                  {[
                    { l: 'Destinations', v: '180+' },
                    { l: 'Elite Agents', v: '24/7' },
                    { l: 'Trust Score', v: '9.9' }
                  ].map(s => (
                    <div key={s.l} className="group/stat">
                      <div className="text-3xl font-bold text-white mb-1.5 group-hover/stat:text-primary transition-colors duration-300">{s.v}</div>
                      <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest group-hover/stat:text-slate-400 transition-colors duration-300">{s.l}</div>
                    </div>
                  ))}
                </div>
              </motion.div>
            ) : isLoading ? (
              <div key="loading" className="h-full flex flex-col items-center justify-center gap-12 p-8 relative">
                <div className="ambient-glow"></div>
                <div className="relative">
                  <motion.div
                    className="w-36 h-36 rounded-full border-[1.5px] border-primary/10 border-t-primary animate-spin-slow"
                  ></motion.div>
                  <motion.div
                    className="absolute inset-0 flex items-center justify-center"
                    animate={{ scale: [1, 1.1, 1], opacity: [0.7, 1, 0.7] }}
                    transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
                  >
                    <div className="bg-gradient-to-br from-primary/20 to-primary/5 p-4 rounded-2xl">
                      <Diamond className="w-10 h-10 text-primary drop-shadow-[0_0_12px_rgba(164,140,244,0.5)]" />
                    </div>
                  </motion.div>
                  {/* Orbiting dots */}
                  {[0, 1, 2].map(i => (
                    <motion.div
                      key={i}
                      className="absolute w-2 h-2 rounded-full bg-primary"
                      style={{
                        width: 6,
                        height: 6,
                        top: '50%',
                        left: '50%',
                        marginTop: -3,
                        marginLeft: -3,
                      }}
                      animate={{
                        x: [0, 60 * Math.cos((i * 120 * Math.PI) / 180), 0],
                        y: [0, 60 * Math.sin((i * 120 * Math.PI) / 180), 0],
                        opacity: [0, 0.6, 0],
                      }}
                      transition={{
                        duration: 3,
                        repeat: Infinity,
                        delay: i * 0.4,
                        ease: "easeInOut",
                      }}
                    />
                  ))}
                </div>
                <div className="text-center space-y-5">
                  <h3 className="text-4xl font-bold text-white tracking-tight">Curating Excellence</h3>
                  <div className="flex gap-2 justify-center">
                    {[0, 1, 2].map(i => (
                      <motion.div
                        key={i}
                        animate={{ scale: [1, 1.6, 1], opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.3 }}
                        className="w-2 h-2 rounded-full bg-primary shadow-[0_0_8px_rgba(164,140,244,0.5)]"
                      ></motion.div>
                    ))}
                  </div>
                  <p className="text-slate-500 font-medium italic">Our elite travel architects are negotiating routes for {destination}...</p>
                </div>
              </div>
            ) : itinerary ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                key="itinerary"
                className="p-10 lg:p-20 pb-32"
              >
                <div className="max-w-7xl mx-auto grid grid-cols-1 xl:grid-cols-4 gap-16">
                  {/* Left Column (3/4) - Main Itinerary */}
                  <div className="xl:col-span-3 space-y-16">
                    {/* Hero Header */}
                    <div className="pb-16 border-b border-white/10 relative">
                      <div className="absolute -top-10 -left-10 w-40 h-40 bg-primary/5 blur-3xl rounded-full"></div>
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                      >
                        <div className="flex items-center gap-3 text-primary font-bold text-xs tracking-[0.4em] uppercase mb-6">
                          <div className="w-6 h-[1px] bg-primary/50"></div>
                          <Diamond className="w-3.5 h-3.5 text-primary" />
                          Confirmed Itinerary
                        </div>
                        <h2 className="text-6xl md:text-7xl font-bold tracking-tighter mb-6 text-white leading-[1.05]">
                          {itinerary.trip_title}
                        </h2>
                        <p className="text-xl md:text-2xl text-slate-400 leading-relaxed max-w-3xl mb-14 font-light">
                          {itinerary.overview}
                        </p>
                        <div className="flex flex-wrap gap-4 stagger-fade-in">
                          <div className="bg-white/[0.04] border border-white/[0.06] rounded-2xl px-6 py-4 flex items-center gap-4 transition-all duration-300 hover:bg-white/[0.07] hover:border-white/10 hover:shadow-[0_4px_20px_rgba(0,0,0,0.2)] group/stat">
                            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center group-hover/stat:bg-emerald-500/20 transition-colors duration-300">
                              <Tree className="w-5 h-5 text-emerald-400" />
                            </div>
                            <div>
                              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-0.5">Sustainability</div>
                              <div className="text-sm font-bold text-white group-hover/stat:text-primary transition-colors duration-300">Level {itinerary.sustainability_score}%</div>
                            </div>
                          </div>
                          <div className="bg-white/[0.04] border border-white/[0.06] rounded-2xl px-6 py-4 flex items-center gap-4 transition-all duration-300 hover:bg-white/[0.07] hover:border-white/10 hover:shadow-[0_4px_20px_rgba(0,0,0,0.2)] group/stat">
                            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center group-hover/stat:bg-primary/20 transition-colors duration-300">
                              <Zap className="w-5 h-5 text-primary" />
                            </div>
                            <div>
                              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-0.5">Budget Class</div>
                              <div className="text-sm font-bold text-white group-hover/stat:text-primary transition-colors duration-300">{itinerary.price_range}</div>
                            </div>
                          </div>
                          <div className="bg-white/[0.04] border border-white/[0.06] rounded-2xl px-6 py-4 flex items-center gap-4 transition-all duration-300 hover:bg-white/[0.07] hover:border-white/10 hover:shadow-[0_4px_20px_rgba(0,0,0,0.2)] group/stat">
                            <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center group-hover/stat:bg-amber-500/20 transition-colors duration-300">
                              <Calendar className="w-5 h-5 text-amber-400" />
                            </div>
                            <div>
                              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-0.5">Duration</div>
                              <div className="text-sm font-bold text-white group-hover/stat:text-primary transition-colors duration-300">{itinerary.days?.length} Premium Days</div>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    </div>

                    {/* Concierge Quote - Premium Block */}
                    <div className="relative group/concierge">
                      <div className="absolute -inset-4 bg-gradient-to-r from-primary/15 to-secondary/10 blur-3xl opacity-30 group-hover/concierge:opacity-60 transition-opacity duration-700 rounded-3xl"></div>
                      <div className="glass-card-premium p-12 md:p-14 relative z-10">
                        <div className="absolute top-0 right-0 p-8 opacity-[0.03]">
                          <Award className="w-40 h-40" />
                        </div>
                        <div className="flex items-start gap-8 relative z-10">
                          <div className="hidden lg:block">
                            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary to-secondary p-[2px] shadow-[0_0_30px_rgba(164,140,244,0.2)] group-hover/concierge:shadow-[0_0_40px_rgba(164,140,244,0.3)] transition-shadow duration-500">
                              <div className="w-full h-full rounded-full bg-[#0c0e12] flex items-center justify-center">
                                <Award className="w-7 h-7 text-primary" />
                              </div>
                            </div>
                          </div>
                          <div className="flex-1">
                            <div className="text-[10px] font-bold text-primary uppercase tracking-[0.3em] mb-6 flex items-center gap-3">
                              <div className="w-8 h-[1px] bg-primary"></div>
                              Executive Director of Concierge
                            </div>
                            <blockquote className="text-2xl md:text-3xl font-light text-slate-200 leading-[1.7] italic serif">
                              "{itinerary.concierge_note}"
                            </blockquote>
                            <div className="mt-8 flex items-center gap-2 text-primary/40 text-[10px] uppercase tracking-[0.2em] font-bold">
                              <div className="w-4 h-[1px] bg-primary/30"></div>
                              Personal Concierge Curation
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Day Selection & Itinerary Flow */}
                    <div className="space-y-12">
                      {/* Tabs Bar */}
                      <div className="sticky top-0 z-10 py-6 -mx-4 px-4 flex gap-3 overflow-x-auto no-scrollbar" style={{ background: 'linear-gradient(180deg, rgba(7,9,13,0.95) 0%, rgba(7,9,13,0.8) 80%, transparent 100%)' }}>
                        {itinerary.days?.map((day, idx) => (
                          <button
                            key={idx}
                            onClick={() => setActiveTab(idx)}
                            className={`flex-1 min-w-[130px] px-6 py-4.5 rounded-2xl flex flex-col items-center gap-1.5 transition-all duration-300 border tab-pill ${activeTab === idx
                              ? 'tab-pill-active text-white'
                              : 'bg-white/[0.03] border-white/5 text-slate-500 hover:border-white/10 hover:bg-white/[0.06] hover:text-slate-300'
                              }`}
                          >
                            <span className={`text-[10px] font-bold uppercase tracking-widest transition-all duration-300 ${activeTab === idx ? 'text-primary/70' : 'opacity-50'}`}>Day</span>
                            <span className="text-base font-bold">{day.day_number}</span>
                            {activeTab === idx && (
                              <motion.div
                                layoutId="tab-indicator"
                                className="w-6 h-0.5 rounded-full bg-primary mt-0.5"
                              />
                            )}
                          </button>
                        ))}
                      </div>

                      {/* Active Day Content */}
                      <AnimatePresence mode="wait">
                        <motion.div
                          key={activeTab}
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -20 }}
                          className="space-y-12"
                        >
                          <div className="flex items-end justify-between border-b border-white/[0.06] pb-10">
                            <div>
                              <div className="text-xs font-bold text-primary uppercase tracking-[0.3em] mb-4 flex items-center gap-3">
                                <div className="w-6 h-[1px] bg-primary/40"></div>
                                Daily Focus
                              </div>
                              <h3 className="text-4xl md:text-5xl font-bold text-white mb-3 leading-tight">{itinerary.days[activeTab]?.theme}</h3>
                              <p className="text-base md:text-lg text-slate-500 font-medium italic">{itinerary.days[activeTab]?.day_name}</p>
                            </div>
                            <div className="hidden md:block">
                              <div className="flex items-center gap-4 bg-white/[0.04] p-4 rounded-2xl border border-white/[0.06] hover:bg-white/[0.06] transition-colors">
                                <div className="text-right">
                                  <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Tempo</div>
                                  <div className="text-sm font-bold text-white">Curated Fluidity</div>
                                </div>
                                <Activity className="w-5 h-5 text-primary" />
                              </div>
                            </div>
                          </div>

                          <div className="space-y-16 relative">
                            {/* Elegant Timeline Connector */}
                            <div className="absolute left-[34px] top-10 bottom-10 w-[1px] bg-gradient-to-b from-primary via-primary/50 to-transparent"></div>

                            {itinerary.days[activeTab]?.activities.map((act, idx) => (
                              <motion.div
                                initial={{ opacity: 0, y: 30 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: idx * 0.15 }}
                                key={idx}
                                className="relative flex gap-12 group"
                              >
                                {/* Timeline Marker */}
                                <div className="shrink-0 flex flex-col items-center">
                                  <div className="w-16 h-16 rounded-full bg-gradient-to-br from-[#15181e] to-[#0c0e12] border border-white/10 flex items-center justify-center relative z-10 group-hover:border-primary/40 transition-all duration-500 timeline-dot">
                                    <div className="w-3 h-3 rounded-full bg-primary shadow-[0_0_20px_#a48cf4] group-hover:shadow-[0_0_30px_#a48cf4] transition-shadow duration-500"></div>
                                  </div>
                                  <div className="mt-3 text-[11px] font-bold text-primary tracking-[0.15em] uppercase bg-primary/10 px-3 py-1 rounded-lg">{act.time}</div>
                                </div>

                                <div className="flex-1 space-y-6">
                                  <div className="space-y-3">
                                    <div className="flex items-start gap-3 flex-wrap">
                                      <h4 className="text-2xl md:text-3xl font-bold text-white group-hover:text-primary transition-colors duration-500 leading-tight">{act.title}</h4>
                                      <div className="px-3.5 py-1.5 rounded-full bg-gradient-to-r from-primary/15 to-primary/5 text-primary text-[10px] font-bold uppercase tracking-widest border border-primary/20 whitespace-nowrap">
                                        {act.tag}
                                      </div>
                                    </div>
                                    <div className="flex items-center gap-2 text-slate-500 text-sm font-medium">
                                      <MapPin className="w-3.5 h-3.5 text-slate-400" />
                                      {act.location}
                                    </div>
                                  </div>

                                  <p className="text-lg md:text-xl text-slate-400 font-light leading-relaxed max-w-2xl">
                                    {act.description}
                                  </p>

                                  <div className="flex gap-3 pt-1">
                                    <a
                                      href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(act.map_query || act.location)}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="bg-white/[0.05] hover:bg-primary/10 p-3 rounded-xl border border-white/[0.06] hover:border-primary/30 text-slate-300 transition-all duration-300 flex items-center gap-2 text-xs font-semibold group/btn"
                                    >
                                      <Navigation className="w-4 h-4 text-primary group-hover/btn:scale-110 transition-transform duration-300" />
                                      Navigate
                                    </a>
                                    <button className="bg-white/[0.05] hover:bg-white/10 p-3 rounded-xl border border-white/[0.06] hover:border-white/10 text-slate-300 transition-all duration-300 flex items-center gap-2 text-xs font-semibold">
                                      <Camera className="w-4 h-4 text-primary" />
                                      Inspiration
                                    </button>
                                  </div>

                                  {act.transport_to_next && (
                                    <div className="glass-card-premium p-8 mt-10 relative overflow-hidden group/trans">
                                      <div className="absolute top-0 right-0 w-40 h-40 bg-primary/[0.04] blur-3xl -mr-10 -mt-10 group-hover/trans:bg-primary/[0.08] transition-all duration-700"></div>
                                      <div className="flex flex-col md:flex-row md:items-center justify-between gap-8 relative z-10">
                                        <div className="flex items-center gap-6">
                                          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary/15 to-primary/5 flex items-center justify-center border border-primary/20 group-hover/trans:border-primary/30 transition-all duration-300">
                                            {act.transport_to_next.mode.toLowerCase().includes('walk') ? <Navigation className="w-7 h-7 text-primary" /> : <Bus className="w-7 h-7 text-primary" />}
                                          </div>
                                          <div>
                                            <div className="text-[10px] font-bold text-primary uppercase tracking-[0.3em] mb-1.5">Transfer</div>
                                            <div className="font-bold text-white text-xl flex items-center gap-2">
                                              {act.transport_to_next.mode}
                                              <ArrowRight className="w-4 h-4 text-slate-600" />
                                            </div>
                                          </div>
                                        </div>
                                        <div className="flex gap-8 md:gap-12">
                                          <div>
                                            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Duration</div>
                                            <div className="font-bold text-white text-base flex items-center gap-2">
                                              <Clock className="w-4 h-4 text-primary" />
                                              {act.transport_to_next.duration}
                                            </div>
                                          </div>
                                          <div>
                                            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Cost</div>
                                            <div className="font-bold text-white text-base flex items-center gap-2">
                                              <Wallet className="w-4 h-4 text-primary" />
                                              {act.transport_to_next.cost}
                                            </div>
                                          </div>
                                        </div>
                                      </div>
                                      <div className="mt-6 pt-6 border-t border-white/[0.05] text-slate-400 text-sm leading-relaxed italic flex items-start gap-3">
                                        <span className="text-primary/30 text-lg leading-none">"</span>
                                        <span>{act.transport_to_next.instructions}</span>
                                        <span className="text-primary/30 text-lg leading-none self-end">"</span>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </motion.div>
                            ))}
                          </div>
                        </motion.div>
                      </AnimatePresence>
                    </div>
                  </div>

                  {/* Right Column (1/4) - Insights & Intelligence */}
                  <div className="xl:col-span-1 space-y-10">
                    {/* Climate Outlook - Premium Widget */}
                    <div className="glass-card overflow-hidden group/climate relative">
                      <div className="absolute top-0 right-0 w-40 h-40 bg-amber-400/5 blur-3xl -mr-12 -mt-12 group-hover/climate:bg-amber-400/10 transition-all duration-700"></div>
                      <div className="absolute inset-x-0 top-0 h-[1px] bg-gradient-to-r from-transparent via-amber-400/20 to-transparent"></div>
                      <div className="p-8">
                        <h3 className="text-xs font-bold text-white mb-6 flex items-center gap-3 tracking-[0.2em] uppercase">
                          <div className="bg-amber-400/10 p-2 rounded-lg">
                            <Sun className="w-4 h-4 text-amber-400" />
                          </div>
                          Climate Outlook
                        </h3>
                        {weather ? (
                          <div className="space-y-6 relative z-10">
                            <div className="flex items-center gap-5">
                              <div className="text-5xl font-bold text-white tracking-tighter">
                                {weather.temperature_c?.expected_high != null
                                  ? `${Math.round(weather.temperature_c.expected_high)}°C`
                                  : 'N/A'}
                              </div>
                              <div className="h-12 w-[1px] bg-white/10"></div>
                              <div className="text-[10px] font-bold text-slate-500 uppercase leading-relaxed tracking-wider">Peak Temp<br /><span className="text-slate-400">Expected</span></div>
                            </div>
                            <div className="space-y-3">
                              <div className="flex items-center gap-2.5 p-3.5 bg-white/[0.04] rounded-xl border border-white/[0.06] group-hover/climate:border-amber-400/20 transition-all duration-500">
                                <div className="w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_12px_#fbbf24] shrink-0"></div>
                                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wide">{weather.conditions_summary || 'Conditions data pending'}</span>
                              </div>
                              <p className="text-[11px] font-medium text-slate-500 leading-relaxed italic px-1">
                                "{weather.temperature_c?.notes || weather.conditions_summary || "Environmental conditions are optimized for your selected itinerary themes."}"
                              </p>
                            </div>
                          </div>
                        ) : (
                          <div className="py-8 text-center space-y-3">
                            <RefreshCcw className="w-7 h-7 text-primary/30 mx-auto animate-spin-slow" />
                            <p className="text-slate-500 text-[10px] font-bold uppercase tracking-widest italic">Synching Intelligence...</p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Local Expert - Soul Panel */}
                    <div className="glass-card overflow-hidden group/soul relative">
                      <div className="absolute inset-x-0 bottom-0 h-[1px] bg-gradient-to-r from-transparent via-primary/40 to-transparent"></div>
                      <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 blur-3xl -mr-8 -mt-8 group-hover/soul:bg-primary/10 transition-all duration-700"></div>
                      <div className="p-8">
                        <h3 className="text-xs font-bold text-white mb-6 flex items-center gap-3 tracking-[0.2em] uppercase">
                          <div className="bg-primary/10 p-2 rounded-lg">
                            <Award className="w-4 h-4 text-primary" />
                          </div>
                          Local Soul Insight
                        </h3>
                        <div className="relative z-10">
                          <div className="text-5xl text-primary/20 font-serif absolute -top-5 -left-2 italic leading-none select-none">"</div>
                          <p className="text-slate-300 text-[13px] leading-[1.9] mb-8 font-light italic relative z-10 pl-2">
                            {localExpert ? (
                              typeof localExpert === 'string' 
                                ? (localExpert.length > 350 ? localExpert.substring(0, 350) + "..." : localExpert)
                                : (localExpert.summary ? (localExpert.summary.length > 350 ? localExpert.summary.substring(0, 350) + "..." : localExpert.summary) : "Loading...")
                            ) : (
                              "We are gathering contemporary cultural nuances and heritage secrets for this specific destination to enhance your perspective."
                            )}
                          </p>
                          <button 
                            onClick={() => setIsIntelligenceOpen(true)}
                            className="w-full py-3.5 rounded-xl bg-white/[0.04] border border-white/[0.06] text-[11px] font-bold text-slate-400 flex items-center justify-center gap-2 hover:bg-white/[0.08] hover:text-white hover:border-primary/20 transition-all duration-300 uppercase tracking-[0.15em] group/btn cursor-pointer"
                          >
                            EXPAND INTELLIGENCE <ChevronRight className="w-3 h-3 text-primary group-hover/btn:translate-x-0.5 transition-transform duration-300" />
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* Mobility Strategy */}
                    <div className="glass-card overflow-hidden group/mob border-white/5">
                      <div className="p-8 pb-2">
                        <h3 className="text-xs font-bold text-white mb-6 flex items-center gap-3 tracking-[0.2em] uppercase">
                          <div className="bg-primary/10 p-2 rounded-lg">
                            <Navigation className="w-4 h-4 text-primary" />
                          </div>
                          Mobility Strategy
                        </h3>
                      </div>

                      <div className="space-y-[1px]">
                        {[
                          { id: 'flights', label: 'Aerial Routes & Logistics', icon: <Plane className="w-4 h-4" />, data: mobility?.flights },
                          { id: 'trains', label: 'Regional Rail Networks', icon: <Train className="w-4 h-4" />, data: mobility?.regional_trains_buses },
                          { id: 'cars', label: 'Private Chauffeur & Hire', icon: <Car className="w-4 h-4" />, data: mobility?.car_rentals },
                          { id: 'airport', label: 'Protocol Transfers', icon: <ShieldCheck className="w-4 h-4" />, data: mobility?.airport_transfers },
                          { id: 'local', label: 'Urban Mobility Protocol', icon: <Bus className="w-4 h-4" />, data: mobility?.local_transport },
                        ].map((item) => (
                          <div key={item.id} className="relative group/item overflow-hidden border-b border-white/[0.03] last:border-0">
                            <button
                              onClick={() => setExpandedMobility(expandedMobility === item.id ? null : item.id)}
                              className={`w-full flex items-center justify-between p-5 bg-[#0c0e12] transition-all duration-300 ${expandedMobility === item.id ? 'bg-[#15181e]' : 'hover:bg-[#15181e]'}`}
                            >
                              <div className="flex items-center gap-4">
                                <div className={`p-2 rounded-lg transition-all duration-300 ${expandedMobility === item.id ? 'bg-primary/15 text-primary' : 'text-slate-500 group-hover/item:text-primary'}`}>
                                  {item.icon}
                                </div>
                                <span className={`text-xs font-semibold tracking-wide transition-all duration-300 ${expandedMobility === item.id ? 'text-white' : 'text-slate-400 group-hover/item:text-slate-200'}`}>
                                  {item.label}
                                </span>
                              </div>
                              <ChevronRight className={`w-4 h-4 transition-all duration-300 ${expandedMobility === item.id ? 'rotate-90 text-primary' : 'text-slate-600 group-hover/item:text-primary group-hover/item:translate-x-0.5'}`} />
                            </button>
                            <AnimatePresence>
                              {expandedMobility === item.id && (
                                <motion.div
                                  initial={{ height: 0 }}
                                  animate={{ height: 'auto' }}
                                  exit={{ height: 0 }}
                                  className="overflow-hidden bg-[#0c0e12]"
                                >
                                  <div className="px-5 pb-5 text-[11px] text-slate-400 space-y-4 font-medium leading-relaxed border-t border-white/[0.04] pt-4">
                                    {item.data ? (
                                      <>
                                        {typeof item.data === 'string' ? (
                                          <div>{item.data}</div>
                                        ) : (
                                          <div className="space-y-4">
                                            {item.data.comparison_tips && (
                                              <div className="space-y-1.5">
                                                <div className="text-primary font-bold uppercase tracking-widest text-[9px] mb-2">Comparison Strategy</div>
                                                {item.data.comparison_tips.map((t: string, i: number) => <div key={i} className="flex gap-2 text-slate-500"><span className="text-primary">•</span> {t}</div>)}
                                              </div>
                                            )}
                                            {item.data.options && (
                                              <div className="space-y-3">
                                                <div className="text-primary font-bold uppercase tracking-widest text-[9px]">Validated Providers</div>
                                                {item.data.options.slice(0, 3).map((o: any, i: number) => (
                                                  <div key={i} className="p-3 bg-white/[0.03] rounded-xl border border-white/5 hover:border-primary/20 transition-all duration-300">
                                                    <div className="font-bold text-white mb-1 text-xs">{o.company || o.mode}</div>
                                                    <div className="text-slate-500 text-[10px] leading-relaxed">{o.pros_cons || o.why || 'No additional details'}</div>
                                                  </div>
                                                ))}
                                              </div>
                                            )}
                                            {!item.data.comparison_tips && !item.data.options && <div className="text-slate-500">Standard protocols apply. Consult our private concierge for detailed routes.</div>}
                                          </div>
                                        )}
                                      </>
                                    ) : (
                                      <div className="flex items-center gap-2 text-slate-600">
                                        <Info className="w-3.5 h-3.5" />
                                        <span>Awaiting synchronized data...</span>
                                      </div>
                                    )}
                                  </div>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        ))}
                      </div>

                      {/* Final Routing Strategy */}
                      <div className="p-8 pt-4 pb-8 space-y-5">
                        <div className="flex flex-col gap-4">
                          <div className="p-5 bg-gradient-to-br from-primary/[0.05] to-primary/[0.02] rounded-2xl border border-primary/20 hover:border-primary/30 transition-all duration-300">
                            <div className="flex items-center gap-2 mb-3">
                              <div className="w-5 h-5 rounded-lg bg-primary/20 flex items-center justify-center">
                                <Zap className="w-3 h-3 text-primary" />
                              </div>
                              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Coordinated Logic</span>
                            </div>
                            <p className="text-[11px] text-slate-400 leading-relaxed font-light italic">
                              {mobility?.route_optimization?.strategy || "Our agents have calculated the most efficient grouping of destinations to minimize transit fatigue."}
                            </p>
                          </div>
                          <button className="w-full bg-gradient-to-r from-primary/[0.08] to-primary/[0.03] hover:from-primary/[0.15] hover:to-primary/[0.08] text-primary py-3.5 rounded-xl text-[10px] font-bold tracking-[0.2em] uppercase transition-all duration-300 flex items-center justify-center gap-2.5 border border-primary/10 hover:border-primary/30 group/btn">
                            <Navigation className="w-4 h-4 group-hover/btn:scale-110 transition-transform duration-300" />
                            ACCESS LIVE PLOT
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            ) : null}
          </AnimatePresence>

          {/* Premium Cultural Intelligence Modal */}
          <AnimatePresence>
            {isIntelligenceOpen && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.4 }}
                className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-lg"
              >
                <motion.div
                  initial={{ scale: 0.94, y: 24, opacity: 0 }}
                  animate={{ scale: 1, y: 0, opacity: 1 }}
                  exit={{ scale: 0.94, y: 24, opacity: 0 }}
                  transition={{ type: 'spring', damping: 28, stiffness: 260 }}
                  className="relative w-full max-w-4xl max-h-[85vh] overflow-y-auto glass-card-premium p-8 md:p-12 shadow-[0_0_80px_rgba(164,140,244,0.12)] flex flex-col gap-8 custom-scrollbar text-left"
                >
                  {/* Header */}
                  {/* Decorative header accent */}
                  <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-primary/40 via-accent/40 to-primary/20"></div>
                  
                  <div className="flex justify-between items-start gap-4">
                    <div>
                      <span className="text-[10px] font-bold text-primary tracking-[0.3em] uppercase flex items-center gap-2">
                        <Diamond className="w-3 h-3 text-primary" />
                        Deep Intelligence Brief
                      </span>
                      <h2 className="text-2xl md:text-3xl font-extrabold text-white mt-2 tracking-tight">
                        {destination || "Destination"} <span className="text-gradient-primary">Living Identity</span>
                      </h2>
                    </div>
                    <button 
                      onClick={() => setIsIntelligenceOpen(false)}
                      className="p-2.5 rounded-xl bg-white/[0.05] border border-white/[0.08] text-slate-400 hover:text-white hover:bg-white/10 hover:border-white/20 transition-all duration-300 cursor-pointer group/close"
                    >
                      <X className="w-4 h-4 group-hover/close:scale-110 transition-transform duration-300" />
                    </button>
                  </div>

                  {/* Content summary */}
                  <p className="text-slate-300 text-sm md:text-base leading-relaxed italic border-l-2 border-primary/40 pl-5 py-2 bg-white/[0.02] rounded-r-xl">
                    "{localExpert?.summary || "Deep heritage insights pending synch."}"
                  </p>

                  {/* Grid of details */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    
                    {/* Contemporary Behaviors */}
                    <div className="p-6 bg-[#0c0e12] border border-white/[0.06] rounded-2xl space-y-4 hover:border-primary/30 transition-all duration-300 group border-shimmer">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors duration-300">
                          <span className="w-2 h-2 rounded-full bg-primary shadow-[0_0_10px_#a48cf4]"></span>
                        </div>
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider">{localExpert?.contemporary_behaviors?.title || "Living Rhythms & Trends"}</h3>
                      </div>
                      <ul className="space-y-2.5">
                        {(localExpert?.contemporary_behaviors?.insights || []).map((ins: string, idx: number) => (
                          <li key={idx} className="text-xs font-medium text-slate-400 leading-relaxed flex items-start gap-2.5 bg-white/[0.02] p-2.5 rounded-lg">
                            <span className="text-primary mt-0.5 shrink-0">•</span>
                            <span>{ins}</span>
                          </li>
                        ))}
                        {(!localExpert?.contemporary_behaviors?.insights || localExpert.contemporary_behaviors.insights.length === 0) && (
                          <li className="text-xs font-medium text-slate-500 italic p-2">No contemporary trends indexed.</li>
                        )}
                      </ul>
                    </div>

                    {/* Unwritten Customs */}
                    <div className="p-6 bg-[#0c0e12] border border-white/[0.06] rounded-2xl space-y-4 hover:border-amber-400/30 transition-all duration-300 group border-shimmer">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="w-8 h-8 rounded-lg bg-amber-400/10 flex items-center justify-center group-hover:bg-amber-400/20 transition-colors duration-300">
                          <span className="w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_10px_#fbbf24]"></span>
                        </div>
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider">{localExpert?.unwritten_customs?.title || "Unwritten Social Codes"}</h3>
                      </div>
                      <ul className="space-y-2.5">
                        {(localExpert?.unwritten_customs?.insights || []).map((ins: string, idx: number) => (
                          <li key={idx} className="text-xs font-medium text-slate-400 leading-relaxed flex items-start gap-2.5 bg-white/[0.02] p-2.5 rounded-lg">
                            <span className="text-amber-400 mt-0.5 shrink-0">•</span>
                            <span>{ins}</span>
                          </li>
                        ))}
                        {(!localExpert?.unwritten_customs?.insights || localExpert.unwritten_customs.insights.length === 0) && (
                          <li className="text-xs font-medium text-slate-500 italic p-2">No social customs indexed.</li>
                        )}
                      </ul>
                    </div>

                    {/* Folklore & Hidden Heritage */}
                    <div className="p-6 bg-[#0c0e12] border border-white/[0.06] rounded-2xl space-y-4 hover:border-emerald-400/30 transition-all duration-300 group border-shimmer">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="w-8 h-8 rounded-lg bg-emerald-400/10 flex items-center justify-center group-hover:bg-emerald-400/20 transition-colors duration-300">
                          <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_10px_#34d399]"></span>
                        </div>
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider">{localExpert?.folklore_heritage?.title || "Folklore & Hidden Heritage"}</h3>
                      </div>
                      <ul className="space-y-2.5">
                        {(localExpert?.folklore_heritage?.insights || []).map((ins: string, idx: number) => (
                          <li key={idx} className="text-xs font-medium text-slate-400 leading-relaxed flex items-start gap-2.5 bg-white/[0.02] p-2.5 rounded-lg">
                            <span className="text-emerald-400 mt-0.5 shrink-0">•</span>
                            <span>{ins}</span>
                          </li>
                        ))}
                        {(!localExpert?.folklore_heritage?.insights || localExpert.folklore_heritage.insights.length === 0) && (
                          <li className="text-xs font-medium text-slate-500 italic p-2">No folklore or heritage stories indexed.</li>
                        )}
                      </ul>
                    </div>

                    {/* Guidebook vs Reality */}
                    <div className="p-6 bg-[#0c0e12] border border-white/[0.06] rounded-2xl space-y-4 hover:border-rose-400/30 transition-all duration-300 group border-shimmer">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="w-8 h-8 rounded-lg bg-rose-400/10 flex items-center justify-center group-hover:bg-rose-400/20 transition-colors duration-300">
                          <span className="w-2 h-2 rounded-full bg-rose-400 shadow-[0_0_10px_#f87171]"></span>
                        </div>
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider">{localExpert?.guidebook_vs_reality?.title || "Guidebook vs. Reality"}</h3>
                      </div>
                      <ul className="space-y-2.5">
                        {(localExpert?.guidebook_vs_reality?.insights || []).map((ins: string, idx: number) => (
                          <li key={idx} className="text-xs font-medium text-slate-400 leading-relaxed flex items-start gap-2.5 bg-white/[0.02] p-2.5 rounded-lg">
                            <span className="text-rose-400 mt-0.5 shrink-0">•</span>
                            <span>{ins}</span>
                          </li>
                        ))}
                        {(!localExpert?.guidebook_vs_reality?.insights || localExpert.guidebook_vs_reality.insights.length === 0) && (
                          <li className="text-xs font-medium text-slate-500 italic p-2">No reality insights indexed.</li>
                        )}
                      </ul>
                    </div>

                    {/* Authenticity Signals */}
                    <div className="p-6 bg-[#0c0e12] border border-white/[0.06] rounded-2xl space-y-4 hover:border-indigo-400/30 transition-all duration-300 group md:col-span-2 border-shimmer">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="w-8 h-8 rounded-lg bg-indigo-400/10 flex items-center justify-center group-hover:bg-indigo-400/20 transition-colors duration-300">
                          <span className="w-2 h-2 rounded-full bg-indigo-400 shadow-[0_0_10px_#818cf8]"></span>
                        </div>
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider">{localExpert?.authenticity_signals?.title || "Living Authenticity Signals"}</h3>
                      </div>
                      <ul className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                        {(localExpert?.authenticity_signals?.insights || []).map((ins: string, idx: number) => (
                          <li key={idx} className="text-xs font-medium text-slate-400 leading-relaxed flex items-start gap-2.5 bg-white/[0.02] p-2.5 rounded-lg">
                            <span className="text-indigo-400 mt-0.5 shrink-0">•</span>
                            <span>{ins}</span>
                          </li>
                        ))}
                        {(!localExpert?.authenticity_signals?.insights || localExpert.authenticity_signals.insights.length === 0) && (
                          <li className="text-xs font-medium text-slate-500 italic md:col-span-2 p-2">No authenticity signals indexed.</li>
                        )}
                      </ul>
                    </div>

                    {/* Sensory Profile */}
                    <div className="p-6 bg-[#0c0e12] border border-white/[0.06] rounded-2xl space-y-4 hover:border-fuchsia-400/30 transition-all duration-300 group md:col-span-2 border-shimmer">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="w-8 h-8 rounded-lg bg-fuchsia-400/10 flex items-center justify-center group-hover:bg-fuchsia-400/20 transition-colors duration-300">
                          <span className="w-2 h-2 rounded-full bg-fuchsia-400 shadow-[0_0_10px_#e879f9]"></span>
                        </div>
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider">{localExpert?.sensory_profile?.title || "Sensory Signature"}</h3>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                        
                        {/* Sounds */}
                        <div className="space-y-3">
                          <span className="text-[10px] font-bold text-fuchsia-400/80 uppercase tracking-widest flex items-center gap-2">
                            <div className="w-3 h-3 rounded bg-fuchsia-400/20 flex items-center justify-center">
                              <span className="w-1 h-1 rounded-full bg-fuchsia-400"></span>
                            </div>
                            Sounds
                          </span>
                          <ul className="space-y-1.5">
                            {(localExpert?.sensory_profile?.sounds || []).map((snd: string, idx: number) => (
                              <li key={idx} className="text-xs font-medium text-slate-400 leading-relaxed flex items-start gap-2 p-2 bg-white/[0.02] rounded-lg">
                                <span className="text-fuchsia-400/50 mt-0.5">♫</span>
                                <span>{snd}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        {/* Scents */}
                        <div className="space-y-3">
                          <span className="text-[10px] font-bold text-fuchsia-400/80 uppercase tracking-widest flex items-center gap-2">
                            <div className="w-3 h-3 rounded bg-fuchsia-400/20 flex items-center justify-center">
                              <span className="w-1 h-1 rounded-full bg-fuchsia-400"></span>
                            </div>
                            Scents
                          </span>
                          <ul className="space-y-1.5">
                            {(localExpert?.sensory_profile?.scents || []).map((sct: string, idx: number) => (
                              <li key={idx} className="text-xs font-medium text-slate-400 leading-relaxed flex items-start gap-2 p-2 bg-white/[0.02] rounded-lg">
                                <span className="text-fuchsia-400/50 mt-0.5">✦</span>
                                <span>{sct}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        {/* Colors */}
                        <div className="space-y-3">
                          <span className="text-[10px] font-bold text-fuchsia-400/80 uppercase tracking-widest flex items-center gap-2">
                            <div className="w-3 h-3 rounded bg-fuchsia-400/20 flex items-center justify-center">
                              <span className="w-1 h-1 rounded-full bg-fuchsia-400"></span>
                            </div>
                            Palette
                          </span>
                          <div className="flex flex-wrap gap-2">
                            {(localExpert?.sensory_profile?.colors || []).map((clr: string, idx: number) => {
                              const hexMatch = clr.match(/#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})/);
                              const colorHex = hexMatch ? hexMatch[0] : '#a48cf4';
                              return (
                                <div key={idx} className="flex items-center gap-2 p-2 bg-white/[0.03] rounded-xl border border-white/5 hover:bg-white/[0.06] hover:border-white/10 transition-all duration-300">
                                  <span 
                                    className="w-3.5 h-3.5 rounded-full border border-white/20" 
                                    style={{ backgroundColor: colorHex }}
                                  ></span>
                                  <span className="text-[10px] font-semibold text-slate-300">{clr}</span>
                                </div>
                              );
                            })}
                          </div>
                        </div>

                      </div>
                    </div>

                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Enhanced Error Overlay */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 100 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 100 }}
                className="fixed bottom-10 left-1/2 -translate-x-1/2 w-full max-w-xl px-4 z-50 flex justify-center"
              >
                <div className="bg-gradient-to-r from-red-500/90 to-rose-500/90 shadow-[0_20px_60px_-15px_rgba(239,68,68,0.4)] text-white px-8 py-5 rounded-2xl flex items-center gap-5 border border-white/10 backdrop-blur-xl">
                  <div className="bg-white/15 p-3 rounded-xl shrink-0">
                    <RefreshCcw className="w-5 h-5 animate-spin-slow" />
                  </div>
                  <div className="flex-1">
                    <p className="font-bold text-[10px] uppercase tracking-[0.2em] opacity-80 mb-0.5">Error</p>
                    <p className="text-sm font-medium leading-tight">{error}</p>
                  </div>
                  <button onClick={() => setError(null)} className="p-2 hover:bg-white/15 rounded-xl transition-all duration-300 shrink-0 group/err">
                    <X className="w-5 h-5 group-hover/err:scale-110 transition-transform duration-300" />
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

export default App;
