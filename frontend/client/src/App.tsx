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
  CheckCircle,
  Zap,
  Award,
  ShieldCheck,
  Info
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

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
  const [localExpert, setLocalExpert] = useState<string | null>(null);
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
      const response = await axios.post(`${API_BASE_URL}/generate-itinerary`, {
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
          <div className="p-8 pb-4 flex items-center gap-3">
            <div className="bg-primary/20 p-2 rounded-xl">
              <Diamond className="w-8 h-8 text-primary shadow-[0_0_15px_rgba(164,140,244,0.4)]" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white">VELURA</h1>
              <p className="text-xs text-slate-500 italic tracking-wide">Personal Travel Architect</p>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Origin (Optional)</label>
              <div className="relative group">
                <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-primary transition-colors" />
                <input
                  type="text"
                  value={origin}
                  onChange={(e) => setOrigin(e.target.value)}
                  placeholder="e.g. New Delhi (DEL)"
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-sm focus:border-primary/50 focus:bg-primary/5 transition-all outline-none"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Destination</label>
              <div className="relative group">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-primary transition-colors" />
                <input
                  type="text"
                  value={destination}
                  onChange={(e) => setDestination(e.target.value)}
                  placeholder="e.g. Kyoto, Japan"
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-sm focus:border-primary/50 focus:bg-primary/5 transition-all outline-none font-medium"
                />
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center px-1">
                <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em]">Duration</label>
                <span className="text-xs font-bold text-primary px-2 py-1 bg-primary/10 rounded-lg">{duration} Days</span>
              </div>
              <input
                type="range"
                min="1"
                max="14"
                value={duration}
                onChange={(e) => setDuration(parseInt(e.target.value))}
                className="w-full accent-primary h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Tier</label>
              <select
                value={budget}
                onChange={(e) => setBudget(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded-xl py-3 px-4 text-sm focus:border-primary/50 bg-none transition-all outline-none appearance-none"
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
                    className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium transition-all border ${selectedInterests.includes(item.id)
                      ? 'bg-primary/20 border-primary/50 text-white'
                      : 'bg-white/5 border-white/5 text-slate-400 hover:bg-white/10 hover:border-white/10'
                      }`}
                  >
                    <span className={selectedInterests.includes(item.id) ? 'text-primary' : 'text-slate-500'}>
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
              className="w-full bg-gradient-to-br from-[#a48cf4] to-[#6e56cf] text-white font-bold py-3.5 px-6 rounded-xl shadow-[0_5px_25px_rgba(164,140,244,0.3)] hover:shadow-[0_8px_35px_rgba(164,140,244,0.5)] hover:-translate-y-0.5 active:translate-y-0 transition-all disabled:opacity-50 disabled:translate-y-0 flex items-center justify-center gap-2 tracking-wide text-xs"
            >
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
              className="w-full bg-white/5 text-slate-400 font-bold py-3.5 px-6 rounded-xl hover:bg-white/10 transition-all text-xs uppercase tracking-widest border border-white/5"
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
                className="h-full flex flex-col items-center justify-center p-8 text-center max-w-3xl mx-auto"
              >
                <div className="relative mb-12">
                  <div className="absolute inset-0 bg-primary blur-[100px] opacity-20 animate-pulse"></div>
                  <motion.div
                    animate={{ y: [0, -10, 0] }}
                    transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                    className="bg-primary/10 p-10 rounded-[2.5rem] border border-primary/20 shadow-2xl relative z-10 backdrop-blur-xl"
                  >
                    <Diamond className="w-20 h-20 text-primary" />
                  </motion.div>
                </div>
                <h2 className="text-7xl font-bold mb-8 text-white leading-tight tracking-tight">Craft Your <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent italic">Bespoke</span> Narrative</h2>
                <p className="text-xl text-slate-400 mb-12 leading-relaxed font-light">
                  Velura transcends standard planning. We curate high-end travel experiences
                  that resonate with your soul and define your legacy.
                </p>
                <div className="grid grid-cols-3 gap-8 w-full max-w-xl mx-auto bg-white/5 p-8 rounded-3xl border border-white/10">
                  {[
                    { l: 'Destinations', v: '180+' },
                    { l: 'Elite Agents', v: '24/7' },
                    { l: 'Trust Score', v: '9.9' }
                  ].map(s => (
                    <div key={s.l}>
                      <div className="text-2xl font-bold text-white mb-1">{s.v}</div>
                      <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{s.l}</div>
                    </div>
                  ))}
                </div>
              </motion.div>
            ) : isLoading ? (
              <div key="loading" className="h-full flex flex-col items-center justify-center gap-10 p-8">
                <div className="relative">
                  <div className="w-32 h-32 border-[2px] border-primary/10 border-t-primary rounded-full animate-spin"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Diamond className="w-10 h-10 text-primary animate-pulse" />
                  </div>
                  {/* Floating particles animation effect could go here */}
                </div>
                <div className="text-center space-y-4">
                  <h3 className="text-3xl font-bold text-white tracking-tight">Curating Excellence</h3>
                  <div className="flex gap-2 justify-center">
                    {[0, 1, 2].map(i => (
                      <motion.div
                        key={i}
                        animate={{ scale: [1, 1.5, 1], opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.3 }}
                        className="w-1.5 h-1.5 rounded-full bg-primary"
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
                        <div className="flex items-center gap-2 text-primary font-bold text-xs tracking-[0.4em] uppercase mb-6">
                          <Diamond className="w-3 h-3" />
                          Confirmed Itinerary
                        </div>
                        <h2 className="text-7xl font-bold tracking-tighter mb-8 text-white">
                          {itinerary.trip_title}
                        </h2>
                        <p className="text-2xl text-slate-400 leading-relaxed max-w-3xl mb-12 font-light">
                          {itinerary.overview}
                        </p>
                        <div className="flex flex-wrap gap-4">
                          <div className="bg-white/5 border border-white/10 rounded-2xl px-6 py-4 flex items-center gap-4 transition-colors hover:bg-white/10">
                            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center">
                              <Tree className="w-5 h-5 text-emerald-400" />
                            </div>
                            <div>
                              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-0.5">Sustainability</div>
                              <div className="text-sm font-bold text-white">Level: {itinerary.sustainability_score}%</div>
                            </div>
                          </div>
                          <div className="bg-white/5 border border-white/10 rounded-2xl px-6 py-4 flex items-center gap-4 transition-colors hover:bg-white/10">
                            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
                              <Zap className="w-5 h-5 text-primary" />
                            </div>
                            <div>
                              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-0.5">Budget Class</div>
                              <div className="text-sm font-bold text-white">{itinerary.price_range}</div>
                            </div>
                          </div>
                          <div className="bg-white/5 border border-white/10 rounded-2xl px-6 py-4 flex items-center gap-4 transition-colors hover:bg-white/10">
                            <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center">
                              <Calendar className="w-5 h-5 text-amber-400" />
                            </div>
                            <div>
                              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-0.5">Duration</div>
                              <div className="text-sm font-bold text-white">{itinerary.days?.length} Premium Days</div>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    </div>

                    {/* Concierge Quote - Premium Block */}
                    <div className="relative group">
                      <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-secondary/20 blur-2xl opacity-50 group-hover:opacity-100 transition-opacity"></div>
                      <div className="glass-card p-12 relative z-10 border-white/10 overflow-hidden">
                        <div className="absolute top-0 right-0 p-8 opacity-5">
                          <Award className="w-32 h-32" />
                        </div>
                        <div className="flex items-start gap-8">
                          <div className="hidden lg:block">
                            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary to-secondary p-[2px]">
                              <div className="w-full h-full rounded-full bg-[#0c0e12] flex items-center justify-center">
                                <Diamond className="w-6 h-6 text-primary" />
                              </div>
                            </div>
                          </div>
                          <div className="flex-1">
                            <div className="text-[10px] font-bold text-primary uppercase tracking-[0.3em] mb-6 flex items-center gap-3">
                              <div className="w-8 h-[1px] bg-primary"></div>
                              Executive Director of Concierge
                            </div>
                            <blockquote className="text-3xl font-light text-slate-200 leading-[1.6] italic">
                              "{itinerary.concierge_note}"
                            </blockquote>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Day Selection & Itinerary Flow */}
                    <div className="space-y-12">
                      {/* Tabs Bar */}
                      <div className="sticky top-0 z-10 py-6 bg-dark/80 backdrop-blur-md -mx-4 px-4 flex gap-3 overflow-x-auto no-scrollbar">
                        {itinerary.days?.map((day, idx) => (
                          <button
                            key={idx}
                            onClick={() => {
                              setActiveTab(idx);
                              // Scroll to day section if needed
                            }}
                            className={`flex-1 min-w-[140px] px-6 py-4 rounded-2xl flex flex-col items-center gap-1 transition-all border ${activeTab === idx
                              ? 'bg-primary border-primary shadow-[0_10px_30px_rgba(164,140,244,0.3)] text-white'
                              : 'bg-white/5 border-white/5 text-slate-500 hover:border-white/10 hover:bg-white/10'
                              }`}
                          >
                            <span className="text-[10px] font-bold uppercase tracking-widest opacity-60">Sequence</span>
                            <span className="text-base font-bold">DAY {day.day_number}</span>
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
                          <div className="flex items-end justify-between border-b border-white/5 pb-10">
                            <div>
                              <div className="text-sm font-bold text-primary uppercase tracking-[0.3em] mb-3">Daily Focus</div>
                              <h3 className="text-5xl font-bold text-white mb-2 leading-tight">{itinerary.days[activeTab]?.theme}</h3>
                              <p className="text-lg text-slate-500 font-medium italic">{itinerary.days[activeTab]?.day_name}</p>
                            </div>
                            <div className="hidden md:block">
                              <div className="flex items-center gap-4 bg-white/5 p-4 rounded-2xl border border-white/10">
                                <div className="text-right">
                                  <div className="text-[10px] font-bold text-slate-500 uppercase">Tempo</div>
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
                                  <div className="w-16 h-16 rounded-full bg-[#0c0e12] border border-white/10 flex items-center justify-center relative z-10 group-hover:border-primary/50 transition-colors duration-500">
                                    <div className="w-2.5 h-2.5 rounded-full bg-primary shadow-[0_0_15px_#a48cf4]"></div>
                                  </div>
                                  <div className="mt-4 text-[10px] font-bold text-primary tracking-widest uppercase">{act.time}</div>
                                </div>

                                <div className="flex-1 space-y-6">
                                  <div className="space-y-2">
                                    <div className="flex items-center gap-3">
                                      <h4 className="text-3xl font-bold text-white group-hover:text-primary transition-colors duration-500">{act.title}</h4>
                                      <div className="px-3 py-1 rounded-full bg-primary/10 text-primary text-[10px] font-bold uppercase tracking-widest">
                                        {act.tag}
                                      </div>
                                    </div>
                                    <div className="flex items-center gap-2 text-slate-500 text-sm font-medium">
                                      <MapPin className="w-3.5 h-3.5" />
                                      {act.location}
                                    </div>
                                  </div>

                                  <p className="text-xl text-slate-400 font-light leading-relaxed max-w-2xl">
                                    {act.description}
                                  </p>

                                  <div className="flex gap-4 pt-2">
                                    <a
                                      href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(act.map_query || act.location)}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="bg-white/5 hover:bg-white/10 p-3 rounded-xl border border-white/10 text-slate-300 transition-all flex items-center gap-2 text-sm font-semibold"
                                    >
                                      <Navigation className="w-4 h-4 text-primary" />
                                      Navigate
                                    </a>
                                    <button className="bg-white/5 hover:bg-white/10 p-3 rounded-xl border border-white/10 text-slate-300 transition-all flex items-center gap-2 text-sm font-semibold">
                                      <Camera className="w-4 h-4 text-primary" />
                                      Inspiration
                                    </button>
                                  </div>

                                  {act.transport_to_next && (
                                    <div className="glass-card border-primary/20 p-8 mt-10 relative overflow-hidden group/trans hover:border-primary/40 transition-all">
                                      <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 blur-3xl -mr-10 -mt-10 group-hover/trans:bg-primary/10"></div>
                                      <div className="flex flex-col md:flex-row md:items-center justify-between gap-8 relative z-10">
                                        <div className="flex items-center gap-6">
                                          <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center border border-primary/20">
                                            {act.transport_to_next.mode.toLowerCase().includes('walk') ? <Navigation className="w-7 h-7 text-primary" /> : <Bus className="w-7 h-7 text-primary" />}
                                          </div>
                                          <div>
                                            <div className="text-[10px] font-bold text-primary uppercase tracking-[0.3em] mb-1">Coordinated Movement</div>
                                            <div className="font-bold text-white text-xl flex items-center gap-2">
                                              {act.transport_to_next.mode}
                                              <ArrowRight className="w-4 h-4 opacity-50" />
                                            </div>
                                          </div>
                                        </div>
                                        <div className="flex gap-10">
                                          <div>
                                            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Transit Time</div>
                                            <div className="font-bold text-white text-lg flex items-center gap-2">
                                              <Clock className="w-4 h-4 text-primary" />
                                              {act.transport_to_next.duration}
                                            </div>
                                          </div>
                                          <div>
                                            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Est. Expense</div>
                                            <div className="font-bold text-white text-lg flex items-center gap-2">
                                              <Wallet className="w-4 h-4 text-primary" />
                                              {act.transport_to_next.cost}
                                            </div>
                                          </div>
                                        </div>
                                      </div>
                                      <div className="mt-6 pt-6 border-t border-white/5 text-slate-400 text-sm leading-relaxed italic">
                                        "{act.transport_to_next.instructions}"
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
                    <div className="glass-card p-10 group hover:border-primary/30 transition-all duration-700 relative overflow-hidden">
                      <div className="absolute top-0 right-0 w-32 h-32 bg-amber-400/5 blur-3xl -mr-10 -mt-10 group-hover:bg-amber-400/10 transition-colors"></div>
                      <h3 className="text-sm font-bold text-white mb-8 flex items-center gap-3 tracking-[0.2em] uppercase">
                        <Sun className="w-5 h-5 text-amber-400" />
                        Climate Outlook
                      </h3>
                      {weather ? (
                        <div className="space-y-8 relative z-10">
                          <div className="flex items-center gap-4">
                            <div className="text-5xl font-bold text-white tracking-tighter">
                              {weather.temperature_c?.expected_high != null
                                ? `${Math.round(weather.temperature_c.expected_high)}°C`
                                : 'N/A'}
                            </div>
                            <div className="h-10 w-[1px] bg-white/10"></div>
                            <div className="text-xs font-bold text-slate-500 uppercase">High Peak<br />Expected</div>
                          </div>
                          <div className="space-y-4">
                            <div className="flex items-center gap-2 p-3 bg-white/5 rounded-xl border border-white/5 group-hover:border-amber-400/20 transition-all">
                              <div className="w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_10px_#fbbf24]"></div>
                              <span className="text-xs font-bold text-slate-300 uppercase underline-offset-4 underline decoration-amber-400/30">{weather.conditions_summary || 'Conditions data pending'}</span>
                            </div>
                            <p className="text-xs font-medium text-slate-500 leading-relaxed font-outfit px-1 italic">
                              "{weather.temperature_c?.notes || weather.conditions_summary || "Environmental conditions are optimized for your selected itinerary themes."}"
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div className="py-10 text-center space-y-3">
                          <RefreshCcw className="w-8 h-8 text-primary/30 mx-auto animate-spin" />
                          <p className="text-slate-500 text-[10px] font-bold uppercase tracking-widest italic">Intelligence Synching...</p>
                        </div>
                      )}
                    </div>

                    {/* Local Expert - Soul Panel */}
                    <div className="glass-card p-10 group overflow-hidden relative border-primary/10">
                      <div className="absolute inset-x-0 bottom-0 h-1 bg-gradient-to-r from-transparent via-primary/40 to-transparent"></div>
                      <h3 className="text-sm font-bold text-white mb-8 flex items-center gap-3 tracking-[0.2em] uppercase">
                        <Award className="w-5 h-5 text-primary" />
                        Local Soul Insight
                      </h3>
                      <div className="relative z-10">
                        <div className="text-4xl text-primary opacity-20 font-serif absolute -top-4 -left-2 italic">"</div>
                        <p className="text-slate-300 text-sm leading-[1.8] mb-8 font-light italic relative z-10">
                          {localExpert ? (
                            localExpert.length > 350 ? localExpert.substring(0, 350) + "..." : localExpert
                          ) : (
                            "We are gathering contemporary cultural nuances and heritage secrets for this specific destination to enhance your perspective."
                          )}
                        </p>
                        <button className="w-full py-4 rounded-xl bg-white/5 border border-white/10 text-xs font-bold text-slate-400 flex items-center justify-center gap-2 hover:bg-white/10 hover:text-white transition-all uppercase tracking-widest">
                          EXPAND INTELLIGENCE <ChevronRight className="w-3 h-3 text-primary" />
                        </button>
                      </div>
                    </div>

                    {/* Mobility Strategy */}
                    <div className="glass-card p-0 overflow-hidden group border-white/5">
                      <div className="p-10 pb-4">
                        <h3 className="text-sm font-bold text-white mb-8 flex items-center gap-3 tracking-[0.2em] uppercase">
                          <Navigation className="w-5 h-5 text-primary" />
                          Mobility Strategy
                        </h3>
                      </div>

                      <div className="space-y-[1px] bg-white/5">
                        {[
                          { id: 'flights', label: 'Aerial Routes & Logistics', icon: <Plane className="w-4 h-4" />, data: mobility?.flights },
                          { id: 'trains', label: 'Regional Rail Networks', icon: <Train className="w-4 h-4" />, data: mobility?.regional_trains_buses },
                          { id: 'cars', label: 'Private Chauffeur & Hire', icon: <Car className="w-4 h-4" />, data: mobility?.car_rentals },
                          { id: 'airport', label: 'Protocol Transfers', icon: <ShieldCheck className="w-4 h-4" />, data: mobility?.airport_transfers },
                          { id: 'local', label: 'Urban Mobility Protocol', icon: <Bus className="w-4 h-4" />, data: mobility?.local_transport },
                        ].map((item) => (
                          <div key={item.id} className="relative group/item overflow-hidden">
                            <button
                              onClick={() => setExpandedMobility(expandedMobility === item.id ? null : item.id)}
                              className={`w-full flex items-center justify-between p-6 bg-[#0c0e12] transition-colors ${expandedMobility === item.id ? 'bg-[#15181e]' : 'hover:bg-[#15181e]'}`}
                            >
                              <div className="flex items-center gap-4">
                                <div className={`${expandedMobility === item.id ? 'text-primary' : 'text-slate-500'} transition-colors group-hover/item:text-primary`}>
                                  {item.icon}
                                </div>
                                <span className={`text-xs font-bold tracking-wide transition-colors ${expandedMobility === item.id ? 'text-white' : 'text-slate-400 group-hover/item:text-slate-200'}`}>
                                  {item.label}
                                </span>
                              </div>
                              <ChevronRight className={`w-4 h-4 text-slate-700 transition-all ${expandedMobility === item.id ? 'rotate-90 text-primary' : 'group-hover/item:text-primary group-hover/item:translate-x-0.5'}`} />
                            </button>
                            <AnimatePresence>
                              {expandedMobility === item.id && (
                                <motion.div
                                  initial={{ height: 0 }}
                                  animate={{ height: 'auto' }}
                                  exit={{ height: 0 }}
                                  className="overflow-hidden bg-[#0c0e12] border-t border-white/5"
                                >
                                  <div className="p-6 text-[11px] text-slate-400 space-y-4 font-medium leading-relaxed">
                                    {item.data ? (
                                      <>
                                        {typeof item.data === 'string' ? (
                                          <div>{item.data}</div>
                                        ) : (
                                          <div className="space-y-4">
                                            {item.data.comparison_tips && (
                                              <div className="space-y-1">
                                                <div className="text-primary font-bold uppercase tracking-widest text-[9px] mb-2">Comparison Strategy</div>
                                                {item.data.comparison_tips.map((t: string, i: number) => <div key={i} className="flex gap-2"><span>•</span> {t}</div>)}
                                              </div>
                                            )}
                                            {item.data.options && (
                                              <div className="space-y-3">
                                                <div className="text-primary font-bold uppercase tracking-widest text-[9px]">Validated Providers</div>
                                                {item.data.options.slice(0, 3).map((o: any, i: number) => (
                                                  <div key={i} className="p-3 bg-white/5 rounded-xl border border-white/5">
                                                    <div className="font-bold text-white mb-1">{o.company || o.mode}</div>
                                                    <div className="opacity-60 text-[9px]">{o.pros_cons || o.why}</div>
                                                  </div>
                                                ))}
                                              </div>
                                            )}
                                            {!item.data.comparison_tips && !item.data.options && <div>Standard protocols apply. Consult our private concierge for detailed routes.</div>}
                                          </div>
                                        )}
                                      </>
                                    ) : (
                                      <div className="flex items-center gap-2 opacity-50">
                                        <Info className="w-3.5 h-3.5" />
                                        Awaiting synchronized data...
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
                      <div className="p-10 pt-4 pb-8 space-y-6">
                        <div className="flex flex-col gap-4">
                          <div className="p-5 bg-primary/5 rounded-2xl border border-primary/20 group-hover:border-primary/40 transition-all">
                            <div className="flex items-center gap-2 mb-3">
                              <Zap className="w-3.5 h-3.5 text-primary" />
                              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Coordinated Logic</span>
                            </div>
                            <p className="text-[11px] text-slate-400 leading-relaxed font-light italic">
                              {mobility?.route_optimization?.strategy || "Our agents have calculated the most efficient grouping of destinations to minimize transit fatigue."}
                            </p>
                          </div>
                          <button className="w-full bg-primary/10 hover:bg-primary/20 text-primary py-4 rounded-xl text-[10px] font-bold tracking-[0.2em] uppercase transition-all flex items-center justify-center gap-2 border border-primary/10">
                            <Navigation className="w-4 h-4 shadow-[0_0_8px_rgba(164,140,244,0.5)]" />
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

          {/* Enhanced Error Overlay */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 100 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 100 }}
                className="fixed bottom-12 left-1/2 -translate-x-1/2 w-full max-w-xl px-4 z-50 flex justify-center"
              >
                <div className="bg-red-500 shadow-[0_20px_60px_-15px_rgba(239,68,68,0.5)] text-white px-8 py-6 rounded-3xl flex items-center gap-5 border border-white/20 backdrop-blur-xl">
                  <div className="bg-white/20 p-3 rounded-2xl">
                    <RefreshCcw className="w-6 h-6 animate-spin-slow" />
                  </div>
                  <div className="mr-4 flex-1">
                    <p className="font-bold text-[10px] uppercase tracking-[0.2em] opacity-80 mb-1">Critical Intelligence Error</p>
                    <p className="text-sm font-medium leading-tight">{error}</p>
                  </div>
                  <button onClick={() => setError(null)} className="p-2 hover:bg-white/10 rounded-xl transition-all">
                    <CheckCircle className="w-7 h-7" />
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
