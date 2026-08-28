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
  X,
  Moon,
  ShoppingBag,
  Dumbbell,
  Music,
  Landmark,
  Globe,
  Shield,
  Sparkles,
  Smile,
  Thermometer
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import AskPlace from './pages/AskPlace';

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
  best_times?: string[];
  activity_suggestions?: string[];
  packing?: string[];
}

const interestsOptions = [
  { id: 'Wellness', icon: <Heart className="w-4 h-4" /> },
  { id: 'Gastronomy', icon: <Utensils className="w-4 h-4" /> },
  { id: 'Photography', icon: <Camera className="w-4 h-4" /> },
  { id: 'History', icon: <History className="w-4 h-4" /> },
  { id: 'Adventure', icon: <Activity className="w-4 h-4" /> },
  { id: 'Art', icon: <Sun className="w-4 h-4" /> },
  { id: 'Nature & Outdoors', icon: <Tree className="w-4 h-4" /> },
  { id: 'Nightlife', icon: <Moon className="w-4 h-4" /> },
  { id: 'Shopping', icon: <ShoppingBag className="w-4 h-4" /> },
  { id: 'Sports', icon: <Dumbbell className="w-4 h-4" /> },
  { id: 'Architecture', icon: <Landmark className="w-4 h-4" /> },
  { id: 'Music & Festivals', icon: <Music className="w-4 h-4" /> },
  { id: 'Wildlife', icon: <Sparkles className="w-4 h-4" /> },
  { id: 'Spirituality', icon: <Smile className="w-4 h-4" /> }
];

const budgetTiers = ["Essential", "Premier", "Elite", "Legendary"];

function App() {
  const [destination, setDestination] = useState('');
  const [origin, setOrigin] = useState('');
  const [duration, setDuration] = useState(3);
  const [budget, setBudget] = useState('Premier');
  const [selectedInterests, setSelectedInterests] = useState(['Wellness', 'Gastronomy']);
  const [travelDates, setTravelDates] = useState('');
  const [groupSize, setGroupSize] = useState(2);
  const [groupType, setGroupType] = useState('Couple');
  const [dietaryRequirements, setDietaryRequirements] = useState<string[]>([]);
  const [accessibility, setAccessibility] = useState<string[]>([]);
  const [pace, setPace] = useState('Moderate');
  const [accommodationPreference, setAccommodationPreference] = useState('No preference');
  const [occasion, setOccasion] = useState('');
  const [languagePreference, setLanguagePreference] = useState('English only');
  const [riskTolerance, setRiskTolerance] = useState('Balanced');
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [itinerary, setItinerary] = useState<ItineraryData | null>(null);
  const [mobility, setMobility] = useState<MobilityData | null>(null);
  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [localExpert, setLocalExpert] = useState<any>(null);
  const [isIntelligenceOpen, setIsIntelligenceOpen] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [expandedMobility, setExpandedMobility] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'itinerary' | 'ask'>('itinerary');

  const toggleInterest = (id: string) => {
    setSelectedInterests(prev =>
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
    if (fieldErrors.interests) setFieldErrors(prev => { const n = {...prev}; delete n.interests; return n; });
  };

  const toggleDietary = (id: string) => {
    setDietaryRequirements(prev =>
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
    if (fieldErrors.dietaryRequirements) setFieldErrors(prev => { const n = {...prev}; delete n.dietaryRequirements; return n; });
  };

  const toggleAccessibility = (id: string) => {
    setAccessibility(prev =>
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
    if (fieldErrors.accessibility) setFieldErrors(prev => { const n = {...prev}; delete n.accessibility; return n; });
  };

  const VALID_INTERESTS = [
    'Wellness', 'Gastronomy', 'Photography', 'History', 'Adventure', 'Art',
    'Nature & Outdoors', 'Nightlife', 'Shopping', 'Sports', 'Architecture',
    'Music & Festivals', 'Wildlife', 'Spirituality'
  ];
  const VALID_PACES = ['Relaxed', 'Moderate', 'Active', 'Intense'];
  const VALID_GROUP_TYPES = ['Solo', 'Couple', 'Family', 'Friends', 'Business'];
  const VALID_BUDGETS = ['Essential', 'Premier', 'Elite', 'Legendary'];
  const VALID_RISK_TOLERANCES = ['Conservative', 'Balanced', 'Adventurous'];
  const VALID_OCCASIONS = ['', 'Honeymoon', 'Birthday', 'Anniversary', 'Graduation', 'Proposal', 'Retirement', 'Festival/Celebration'];
  const VALID_ACCOMMODATION = ['No preference', 'Hotel', 'Hostel', 'Airbnb/Vacation Rental', 'Boutique/Heritage Stay', 'Camping/Glamping', 'Luxury Resort'];
  const VALID_LANGUAGE = ['English only', 'Basic local phrases', 'Conversational local', 'Fluent local'];
  const VALID_DIETARY = ['Vegetarian', 'Vegan', 'Halal', 'Kosher', 'Gluten-free', 'Nut allergy', 'Lactose intolerant', 'No restrictions'];
  const VALID_ACCESSIBILITY = ['Wheelchair', 'Limited mobility', 'Stroller-friendly', 'Visual support', 'Hearing support', 'None'];

  const validateFields = (): boolean => {
    const errors: Record<string, string> = {};

    if (!destination || !destination.trim()) {
      errors.destination = 'Destination is required.';
    }
    if (!VALID_PACES.includes(pace)) {
      errors.pace = 'Please select a valid pace.';
    }
    if (!VALID_GROUP_TYPES.includes(groupType)) {
      errors.groupType = 'Please select a valid group type.';
    }
    if (duration < 1 || duration > 14) {
      errors.duration = 'Duration must be between 1 and 14 days.';
    }
    if (groupSize < 1 || groupSize > 12) {
      errors.groupSize = 'Group size must be between 1 and 12.';
    }
    if (selectedInterests.length === 0) {
      errors.interests = 'Select at least one interest.';
    }
    const invalidInterests = selectedInterests.filter(i => !VALID_INTERESTS.includes(i));
    if (invalidInterests.length > 0) {
      errors.interests = `Invalid interests: ${invalidInterests.join(', ')}`;
    }
    if (!VALID_BUDGETS.includes(budget)) {
      errors.budget = 'Please select a valid budget tier.';
    }
    if (!VALID_RISK_TOLERANCES.includes(riskTolerance)) {
      errors.riskTolerance = 'Please select a valid exploration style.';
    }
    if (occasion && !VALID_OCCASIONS.includes(occasion)) {
      errors.occasion = 'Please select a valid occasion.';
    }
    if (!VALID_ACCOMMODATION.includes(accommodationPreference)) {
      errors.accommodationPreference = 'Please select a valid accommodation option.';
    }
    if (!VALID_LANGUAGE.includes(languagePreference)) {
      errors.languagePreference = 'Please select a valid language preference.';
    }
    const invalidDietary = dietaryRequirements.filter(d => !VALID_DIETARY.includes(d));
    if (invalidDietary.length > 0) {
      errors.dietaryRequirements = `Invalid dietary options: ${invalidDietary.join(', ')}`;
    }
    const invalidAccessibility = accessibility.filter(a => !VALID_ACCESSIBILITY.includes(a));
    if (invalidAccessibility.length > 0) {
      errors.accessibility = `Invalid accessibility options: ${invalidAccessibility.join(', ')}`;
    }

    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const fieldErrorClass = (field: string) => fieldErrors[field]
    ? 'border-red-500/60 focus:border-red-500/80 focus:bg-red-500/[0.06] focus:shadow-[0_0_20px_rgba(239,68,68,0.08)]'
    : '';

  const InlineError: React.FC<{ field: string }> = ({ field }) => {
    if (!fieldErrors[field]) return null;
    return (
      <p className="text-[10px] font-medium text-red-400 mt-1.5 ml-1 flex items-center gap-1">
        <span className="text-red-400/60">⚠</span> {fieldErrors[field]}
      </p>
    );
  };

  const handleGenerate = async () => {
    if (!validateFields()) {
      return;
    }

    setIsLoading(true);
    setError(null);
    setFieldErrors({});
    setItinerary(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/generate-itinerary`, {
        origin,
        destination,
        duration,
        budget,
        interests: selectedInterests,
        travel_dates: travelDates,
        group_size: groupSize,
        group_type: groupType,
        dietary_requirements: dietaryRequirements,
        accessibility: accessibility,
        pace: pace,
        accommodation_preference: accommodationPreference,
        occasion: occasion,
        language_preference: languagePreference,
        risk_tolerance: riskTolerance
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
      const detail = err.response?.data?.detail || '';
      if (detail.includes('capacity') || detail.includes('rate limit') || detail.includes('overloaded')) {
        setError("Travel planning is temporarily busy. Please try again in a moment.");
      } else if (detail) {
        setError(detail);
      } else if (err.code === 'ECONNREFUSED' || err.code === 'ERR_NETWORK') {
        setError("Cannot reach the server. Please check your connection and try again.");
      } else {
        setError("Something went wrong while planning your trip. Please try again.");
      }
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
    setTravelDates('');
    setGroupSize(2);
    setGroupType('Couple');
    setDietaryRequirements([]);
    setAccessibility([]);
    setPace('Moderate');
    setAccommodationPreference('No preference');
    setOccasion('');
    setLanguagePreference('English only');
    setRiskTolerance('Balanced');
    setError(null);
    setFieldErrors({});
    setExpandedMobility(null);
  };

  const handleResetAndSwitch = (mode: 'itinerary' | 'ask') => {
    if (mode === 'itinerary') {
      handleReset();
    }
    setViewMode(mode);
  };

  return (
    <div className="main-gradient min-h-screen font-outfit text-slate-200">
      {/* Floating particle stars */}
      <div className="stars-container" aria-hidden="true">
        {Array.from({ length: 50 }).map((_, i) => (
          <div
            key={i}
            className="star"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              width: `${Math.random() * 2.5 + 1}px`,
              height: `${Math.random() * 2.5 + 1}px`,
              animationDelay: `${Math.random() * 15}s`,
              animationDuration: `${Math.random() * 10 + 10}s`,
              background: ['#38bdf8', '#2dd4bf', '#fbbf24', '#fb7185', '#e879f9', '#ffffff'][Math.floor(Math.random() * 6)],
              opacity: Math.random() * 0.5 + 0.1,
            }}
          />
        ))}
      </div>

      {/* Extra ambient glowing orbs */}
      <div className="glow-orb glow-orb--primary" style={{ top: '15%', left: '-10%' }}></div>
      <div className="glow-orb glow-orb--accent" style={{ top: '60%', right: '-5%' }}></div>
      <div className="glow-orb glow-orb--amber" style={{ bottom: '10%', left: '30%' }}></div>

      <div className="flex h-screen overflow-hidden">
        {/* Sidebar */}
        <aside className="w-80 bg-[#0c0e12] border-r border-white/5 flex flex-col shrink-0 z-20 relative">
          {/* Sidebar gradient accent line */}
          <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-primary via-teal via-amber to-rose opacity-60"></div>
          <div className="p-8 pb-4 flex items-center gap-4 relative">
            <div className="absolute inset-x-6 -bottom-2 h-[1px] bg-gradient-to-r from-transparent via-primary/30 via-teal/20 via-amber/20 to-transparent"></div>
            <div className="bg-gradient-to-br from-primary/25 via-teal/15 to-secondary/20 p-3 rounded-2xl border border-primary/10 shadow-[0_0_30px_rgba(56,189,248,0.2)]">
              <Diamond className="w-7 h-7 text-primary drop-shadow-[0_0_8px_rgba(56,189,248,0.5)]" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white">
                <span className="text-gradient-shimmer">XPLORA</span>
              </h1>
              <p className="text-[10px] text-slate-500 italic tracking-[0.15em] mt-0.5">Intelligent Travel Architect</p>
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
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pl-10 pr-4 text-sm focus:border-primary/50 focus:bg-primary/[0.06] focus:shadow-[0_0_20px_rgba(56,189,248,0.06)] transition-all duration-300 outline-none"
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
                  onChange={(e) => {
                    setDestination(e.target.value);
                    if (fieldErrors.destination) setFieldErrors(prev => { const n = {...prev}; delete n.destination; return n; });
                  }}
                  placeholder="e.g. Kyoto, Japan"
                  className={`w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pl-10 pr-4 text-sm focus:border-primary/50 focus:bg-primary/[0.06] focus:shadow-[0_0_20px_rgba(56,189,248,0.06)] transition-all duration-300 outline-none font-medium ${fieldErrorClass('destination')}`}
                />
                <InlineError field="destination" />
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center px-1">
                <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em]">Duration</label>
                <span className="text-xs font-bold text-primary px-3 py-1.5 bg-gradient-to-r from-primary/15 to-primary/5 rounded-lg border border-primary/10 shadow-[0_0_15px_rgba(56,189,248,0.08)]">{duration} Days</span>
              </div>
              <input
                type="range"
                min="1"
                max="14"
                value={duration}
                onChange={(e) => {
                  setDuration(parseInt(e.target.value));
                  if (fieldErrors.duration) setFieldErrors(prev => { const n = {...prev}; delete n.duration; return n; });
                }}
                className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer"
              />
              <InlineError field="duration" />
            </div>

            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Tier</label>
              <select
                value={budget}
                onChange={(e) => {
                  setBudget(e.target.value);
                  if (fieldErrors.budget) setFieldErrors(prev => { const n = {...prev}; delete n.budget; return n; });
                }}
                className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 px-4 text-sm focus:border-primary/50 transition-all duration-300 outline-none appearance-none cursor-pointer hover:bg-white/[0.07]"
              >
                {budgetTiers.map(tier => (
                  <option key={tier} value={tier} className="bg-[#0c0e12]">{tier}</option>
                ))}
              </select>
              <InlineError field="budget" />
            </div>

            <div className="space-y-3">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Focus</label>
              <div className="grid grid-cols-2 gap-2">
                {interestsOptions.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => toggleInterest(item.id)}
                    className={`flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all duration-300 border ${selectedInterests.includes(item.id)
                      ? 'bg-gradient-to-br from-primary/25 to-primary/10 border-primary/50 text-white shadow-[0_0_20px_rgba(56,189,248,0.1)] hover:shadow-[0_0_30px_rgba(56,189,248,0.2)]'
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
              <InlineError field="interests" />
            </div>

            {/* Divider */}
            <div className="h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>

            {/* Travel Dates */}
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Travel Dates</label>
              <div className="relative group input-glow rounded-xl">
                <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-primary transition-colors duration-300" />
                <input
                  type="text"
                  value={travelDates}
                  onChange={(e) => setTravelDates(e.target.value)}
                  placeholder="e.g. Spring 2026, Dec 15-22"
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pl-10 pr-4 text-sm focus:border-primary/50 focus:bg-primary/[0.06] focus:shadow-[0_0_20px_rgba(56,189,248,0.06)] transition-all duration-300 outline-none"
                />
              </div>
            </div>

            {/* Group Size & Type */}
            <div className="space-y-2">
              <div className="flex justify-between items-center px-1">
                <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em]">Group Size</label>
                <span className="text-xs font-bold text-primary px-3 py-1.5 bg-gradient-to-r from-primary/15 to-primary/5 rounded-lg border border-primary/10 shadow-[0_0_15px_rgba(56,189,248,0.08)]">{groupSize}</span>
              </div>
              <input
                type="range"
                min="1"
                max="12"
                value={groupSize}
                onChange={(e) => {
                  setGroupSize(parseInt(e.target.value));
                  if (fieldErrors.groupSize) setFieldErrors(prev => { const n = {...prev}; delete n.groupSize; return n; });
                }}
                className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer"
              />
              <InlineError field="groupSize" />
              <div className="grid grid-cols-2 gap-2">
                {['Solo', 'Couple', 'Family', 'Friends', 'Business'].map((type) => (
                  <button
                    key={type}
                    onClick={() => {
                      setGroupType(type);
                      if (fieldErrors.groupType) setFieldErrors(prev => { const n = {...prev}; delete n.groupType; return n; });
                    }}
                    className={`px-3 py-2 rounded-xl text-xs font-medium transition-all duration-300 border ${groupType === type
                      ? 'bg-gradient-to-br from-primary/25 to-primary/10 border-primary/50 text-white shadow-[0_0_15px_rgba(56,189,248,0.1)]'
                      : 'bg-white/5 border-white/5 text-slate-400 hover:bg-white/10 hover:border-white/10'
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>

            {/* Divider */}
            <div className="h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>

            {/* Pace */}
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Pace</label>
              <div className="grid grid-cols-2 gap-2">
                {[{ id: 'Relaxed', emoji: '🐢' }, { id: 'Moderate', emoji: '🚶' }, { id: 'Active', emoji: '🏃' }, { id: 'Intense', emoji: '⚡' }].map((p) => (
                  <button
                    key={p.id}
                    onClick={() => {
                      setPace(p.id);
                      if (fieldErrors.pace) setFieldErrors(prev => { const n = {...prev}; delete n.pace; return n; });
                    }}
                    className={`px-3 py-2.5 rounded-xl text-xs font-medium transition-all duration-300 border ${pace === p.id
                      ? 'bg-gradient-to-br from-primary/25 to-primary/10 border-primary/50 text-white shadow-[0_0_15px_rgba(56,189,248,0.1)]'
                      : 'bg-white/5 border-white/5 text-slate-400 hover:bg-white/10 hover:border-white/10'
                    }`}
                  >
                    {p.emoji} {p.id}
                  </button>
                ))}
              </div>
              <InlineError field="pace" />
            </div>

            {/* Divider */}
            <div className="h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>

            {/* Dietary Requirements */}
            <div className="space-y-3">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Dietary Needs</label>
              <div className="grid grid-cols-2 gap-2">
                {['Vegetarian', 'Vegan', 'Halal', 'Kosher', 'Gluten-free', 'Nut allergy', 'Lactose intolerant', 'No restrictions'].map((item) => (
                  <button
                    key={item}
                    onClick={() => toggleDietary(item)}
                    className={`px-3 py-2 rounded-xl text-[11px] font-medium transition-all duration-300 border ${dietaryRequirements.includes(item)
                      ? 'bg-gradient-to-br from-emerald-500/25 to-emerald-500/10 border-emerald-500/50 text-white shadow-[0_0_15px_rgba(52,211,153,0.1)]'
                      : 'bg-white/5 border-white/5 text-slate-400 hover:bg-white/10 hover:border-white/10'
                    }`}
                  >
                    {item}
                  </button>
                ))}
              </div>
              <InlineError field="dietaryRequirements" />
            </div>

            {/* Divider */}
            <div className="h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>

            {/* Accommodation Preference */}
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Accommodation</label>
              <select
                value={accommodationPreference}
                onChange={(e) => {
                  setAccommodationPreference(e.target.value);
                  if (fieldErrors.accommodationPreference) setFieldErrors(prev => { const n = {...prev}; delete n.accommodationPreference; return n; });
                }}
                className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 px-4 text-sm focus:border-primary/50 transition-all duration-300 outline-none appearance-none cursor-pointer hover:bg-white/[0.07]"
              >
                {['No preference', 'Hotel', 'Hostel', 'Airbnb/Vacation Rental', 'Boutique/Heritage Stay', 'Camping/Glamping', 'Luxury Resort'].map((opt) => (
                  <option key={opt} value={opt} className="bg-[#0c0e12]">{opt}</option>
                ))}
              </select>
              <InlineError field="accommodationPreference" />
            </div>

            {/* Divider */}
            <div className="h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>

            {/* Accessibility */}
            <div className="space-y-3">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Accessibility</label>
              <div className="grid grid-cols-2 gap-2">
                {['Wheelchair', 'Limited mobility', 'Stroller-friendly', 'Visual support', 'Hearing support', 'None'].map((item) => (
                  <button
                    key={item}
                    onClick={() => toggleAccessibility(item)}
                    className={`px-3 py-2 rounded-xl text-[11px] font-medium transition-all duration-300 border ${accessibility.includes(item)
                      ? 'bg-gradient-to-br from-indigo-500/25 to-indigo-500/10 border-indigo-500/50 text-white shadow-[0_0_15px_rgba(129,140,248,0.1)]'
                      : 'bg-white/5 border-white/5 text-slate-400 hover:bg-white/10 hover:border-white/10'
                    }`}
                  >
                    {item}
                  </button>
                ))}
              </div>
              <InlineError field="accessibility" />
            </div>

            {/* Divider */}
            <div className="h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>

            {/* Special Occasion */}
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Special Occasion</label>
              <select
                value={occasion}
                onChange={(e) => {
                  setOccasion(e.target.value);
                  if (fieldErrors.occasion) setFieldErrors(prev => { const n = {...prev}; delete n.occasion; return n; });
                }}
                className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 px-4 text-sm focus:border-primary/50 transition-all duration-300 outline-none appearance-none cursor-pointer hover:bg-white/[0.07]"
              >
                {["", "Honeymoon", "Birthday", "Anniversary", "Graduation", "Proposal", "Retirement", "Festival/Celebration"].map((opt) => (
                  <option key={opt} value={opt} className="bg-[#0c0e12]">{opt || 'None'}</option>
                ))}
              </select>
              <InlineError field="occasion" />
            </div>

            {/* Divider */}
            <div className="h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>

            {/* Language Preference */}
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Language</label>
              <select
                value={languagePreference}
                onChange={(e) => {
                  setLanguagePreference(e.target.value);
                  if (fieldErrors.languagePreference) setFieldErrors(prev => { const n = {...prev}; delete n.languagePreference; return n; });
                }}
                className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 px-4 text-sm focus:border-primary/50 transition-all duration-300 outline-none appearance-none cursor-pointer hover:bg-white/[0.07]"
              >
                {['English only', 'Basic local phrases', 'Conversational local', 'Fluent local'].map((opt) => (
                  <option key={opt} value={opt} className="bg-[#0c0e12]">{opt}</option>
                ))}
              </select>
              <InlineError field="languagePreference" />
            </div>

            {/* Divider */}
            <div className="h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>

            {/* Risk Tolerance */}
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Exploration Style</label>
              <div className="grid grid-cols-3 gap-2">
                {[{ id: 'Conservative', icon: <Shield className="w-3.5 h-3.5" />, desc: 'Safe zones' }, { id: 'Balanced', icon: <Globe className="w-3.5 h-3.5" />, desc: 'Mix' }, { id: 'Adventurous', icon: <Sparkles className="w-3.5 h-3.5" />, desc: 'Off-path' }].map((r) => (
                  <button
                    key={r.id}
                    onClick={() => setRiskTolerance(r.id)}
                    className={`flex flex-col items-center gap-1.5 px-2 py-2.5 rounded-xl text-xs font-medium transition-all duration-300 border ${riskTolerance === r.id
                      ? 'bg-gradient-to-br from-amber-500/25 to-amber-500/10 border-amber-500/50 text-white shadow-[0_0_15px_rgba(245,158,11,0.1)]'
                      : 'bg-white/5 border-white/5 text-slate-400 hover:bg-white/10 hover:border-white/10'
                    }`}
                  >
                    <span className={riskTolerance === r.id ? 'text-amber-400' : 'text-slate-500'}>{r.icon}</span>
                    <span className="text-[10px]">{r.id}</span>
                  </button>
                ))}
              </div>
              <InlineError field="riskTolerance" />
            </div>
          </div>

          <div className="p-6 pt-2 space-y-3 border-t border-white/5 mt-auto">
            {/* View mode toggle */}
            <div className="flex gap-2">
              <button
                onClick={() => handleResetAndSwitch('itinerary')}
                className={`flex-1 py-2.5 px-3 rounded-xl text-xs font-bold uppercase tracking-widest transition-all duration-300 border ${viewMode === 'itinerary'
                  ? 'bg-gradient-to-br from-primary/25 to-primary/10 border-primary/50 text-white shadow-[0_0_15px_rgba(56,189,248,0.1)]'
                  : 'bg-white/[0.04] border-white/[0.06] text-slate-400 hover:bg-white/[0.08] hover:text-slate-300'
                }`}
              >
                <span className="flex items-center justify-center gap-1.5">
                  <Diamond className="w-3.5 h-3.5" />
                  Itinerary
                </span>
              </button>
              <button
                onClick={() => handleResetAndSwitch('ask')}
                className={`flex-1 py-2.5 px-3 rounded-xl text-xs font-bold uppercase tracking-widest transition-all duration-300 border ${viewMode === 'ask'
                  ? 'bg-gradient-to-br from-teal/25 to-teal/10 border-teal/50 text-white shadow-[0_0_15px_rgba(45,212,191,0.1)]'
                  : 'bg-white/[0.04] border-white/[0.06] text-slate-400 hover:bg-white/[0.08] hover:text-slate-300'
                }`}
              >
                <span className="flex items-center justify-center gap-1.5">
                  <Search className="w-3.5 h-3.5" />
                  Ask
                </span>
              </button>
            </div>
            {viewMode === 'itinerary' && (
              <>
              <button
                onClick={handleGenerate}
              disabled={isLoading}
              className="w-full text-white font-bold py-4 px-6 rounded-xl shadow-[0_5px_25px_rgba(56,189,248,0.3)] hover:shadow-[0_8px_40px_rgba(56,189,248,0.5)] hover:shadow-[0_0_30px_rgba(45,212,191,0.2)] hover:-translate-y-0.5 active:translate-y-0 transition-all duration-300 disabled:opacity-50 disabled:translate-y-0 flex items-center justify-center gap-2.5 tracking-[0.12em] text-xs relative overflow-hidden group/btn"
              style={{ background: 'linear-gradient(135deg, #38bdf8 0%, #0284c7 25%, #2dd4bf 65%, #0d9488 100%)', backgroundSize: '200% 200%', animation: 'gradient-shift 4s ease-in-out infinite' }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/15 to-transparent opacity-0 group-hover/btn:opacity-100 transition-opacity duration-700 -skew-x-12 translate-x-[-100%] group-hover/btn:translate-x-[100%] duration-1000"></div>
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-teal/10 to-transparent opacity-0 group-hover/btn:opacity-100 transition-opacity duration-700"></div>
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
              className="w-full bg-white/[0.04] text-slate-400 font-bold py-3.5 px-6 rounded-xl hover:bg-white/[0.08] hover:text-slate-300 hover:border-primary/20 transition-all duration-300 text-xs uppercase tracking-widest border border-white/[0.06] hover:border-teal/20 group/reset"
            >
              <span className="group-hover/reset:bg-gradient-to-r group-hover/reset:from-slate-300 group-hover/reset:to-slate-400 inline-block transition-all duration-300">RESET</span>
            </button>
            </>
            )}
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 overflow-y-auto scroll-smooth relative">
          {viewMode === 'ask' ? (
            <AskPlace />
          ) : (
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
                    <Diamond className="w-24 h-24 text-primary relative z-10 drop-shadow-[0_0_20px_rgba(56,189,248,0.4)]" />
                  </motion.div>
                </div>
                
                <h2 className="text-6xl md:text-7xl font-bold mb-6 text-white leading-tight tracking-tight">
                  Craft Your{' '}
                  <span className="text-gradient-rainbow italic">Bespoke</span>
                  {' '}Narrative
                </h2>
                <p className="text-lg md:text-xl text-slate-400 mb-14 leading-relaxed font-light max-w-2xl">
                  Xplora transcends standard planning. We curate intelligent travel experiences
                  that resonate with your soul and define your legacy.
                </p>
                

              </motion.div>
            ) : isLoading ? (
              <div key="loading" className="h-full flex flex-col items-center justify-center gap-12 p-8 relative">
                <div className="ambient-glow"></div>
                <div className="ambient-glow--accent"></div>
                <div className="relative">
                  <motion.div
                    className="w-36 h-36 rounded-full animate-spin-gradient"
                    style={{
                      border: '1.5px solid',
                      borderColor: 'rgba(56,189,248,0.1)',
                      borderTopColor: '#38bdf8',
                      borderRightColor: '#2dd4bf',
                      borderBottomColor: '#fbbf24',
                      borderLeftColor: '#fb7185',
                      boxShadow: '0 0 40px rgba(56,189,248,0.1)',
                    }}
                  ></motion.div>
                  <motion.div
                    className="absolute inset-0 flex items-center justify-center"
                    animate={{ scale: [1, 1.1, 1], opacity: [0.7, 1, 0.7] }}
                    transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
                  >
                    <div className="bg-gradient-to-br from-primary/20 via-teal/10 to-amber/10 p-4 rounded-2xl border border-primary/10">
                      <Diamond className="w-10 h-10 text-primary drop-shadow-[0_0_12px_rgba(56,189,248,0.5)]" />
                    </div>
                  </motion.div>
                  {/* Colorful orbiting dots */}
                  {[
                    { color: '#38bdf8', delay: 0 },
                    { color: '#2dd4bf', delay: 0.4 },
                    { color: '#fbbf24', delay: 0.8 },
                    { color: '#fb7185', delay: 1.2 },
                    { color: '#e879f9', delay: 1.6 },
                  ].map((dot, i) => (
                    <motion.div
                      key={i}
                      className="absolute rounded-full"
                      style={{
                        width: [6, 5, 7, 4, 8][i],
                        height: [6, 5, 7, 4, 8][i],
                        backgroundColor: dot.color,
                        top: '50%',
                        left: '50%',
                        marginTop: -[3, 2.5, 3.5, 2, 4][i],
                        marginLeft: -[3, 2.5, 3.5, 2, 4][i],
                        boxShadow: `0 0 12px ${dot.color}60`,
                      }}
                      animate={{
                        x: [0, 70 * Math.cos((i * 72 * Math.PI) / 180), 0],
                        y: [0, 70 * Math.sin((i * 72 * Math.PI) / 180), 0],
                        opacity: [0, 0.7, 0],
                      }}
                      transition={{
                        duration: 3.5,
                        repeat: Infinity,
                        delay: dot.delay,
                        ease: "easeInOut",
                      }}
                    />
                  ))}
                </div>
                <div className="text-center space-y-5">
                  <h3 className="text-4xl font-bold text-white tracking-tight">
                    <span className="text-gradient-shimmer">Crafting Your Journey</span>
                  </h3>
                  <div className="flex gap-2 justify-center">
                    {[0, 1, 2].map(i => (
                      <motion.div
                        key={i}
                        animate={{ scale: [1, 1.6, 1], opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.3 }}
                        className={`w-2 h-2 rounded-full ${i === 0 ? 'bg-primary' : i === 1 ? 'bg-teal' : 'bg-amber'} shadow-[0_0_8px_rgba(56,189,248,0.5)]`}
                      ></motion.div>
                    ))}
                  </div>
                  <div className="flex items-center gap-3 justify-center">
                    <motion.span 
                      animate={{ width: ['0%', '100%', '0%'] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="h-[1px] bg-gradient-to-r from-primary via-teal to-amber max-w-[100px]"
                    />
                    <span className="text-slate-500 font-medium italic">Our travel architects are researching routes for {destination}...</span>
                    <motion.span 
                      animate={{ width: ['0%', '100%', '0%'] }}
                      transition={{ duration: 2, repeat: Infinity, delay: 1 }}
                      className="h-[1px] bg-gradient-to-r from-amber via-teal to-primary max-w-[100px]"
                    />
                  </div>
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
                          <div className="bg-white/[0.04] border border-white/[0.06] rounded-2xl px-6 py-4 flex items-center gap-4 transition-all duration-300 hover:bg-white/[0.07] hover:border-emerald-500/30 hover:shadow-[0_4px_25px_rgba(52,211,153,0.12)] group/stat">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500/20 to-emerald-500/5 flex items-center justify-center group-hover/stat:from-emerald-500/30 group-hover/stat:to-emerald-500/10 transition-all duration-300 border border-emerald-500/20">
                              <Tree className="w-5 h-5 text-emerald-400 group-hover/stat:scale-110 transition-transform duration-300" />
                            </div>
                            <div>
                              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-0.5">Sustainability</div>
                              <div className="text-sm font-bold text-white group-hover/stat:text-emerald-400 transition-colors duration-300">Level {itinerary.sustainability_score}%</div>
                            </div>
                          </div>
                          <div className="bg-white/[0.04] border border-white/[0.06] rounded-2xl px-6 py-4 flex items-center gap-4 transition-all duration-300 hover:bg-white/[0.07] hover:border-primary/30 hover:shadow-[0_4px_25px_rgba(56,189,248,0.12)] group/stat">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center group-hover/stat:from-primary/30 group-hover/stat:to-primary/10 transition-all duration-300 border border-primary/20">
                              <Zap className="w-5 h-5 text-primary group-hover/stat:scale-110 transition-transform duration-300" />
                            </div>
                            <div>
                              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-0.5">Budget Class</div>
                              <div className="text-sm font-bold text-white group-hover/stat:text-primary transition-colors duration-300">{itinerary.price_range}</div>
                            </div>
                          </div>
                          <div className="bg-white/[0.04] border border-white/[0.06] rounded-2xl px-6 py-4 flex items-center gap-4 transition-all duration-300 hover:bg-white/[0.07] hover:border-amber-500/30 hover:shadow-[0_4px_25px_rgba(245,158,11,0.12)] group/stat">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500/20 to-amber-500/5 flex items-center justify-center group-hover/stat:from-amber-500/30 group-hover/stat:to-amber-500/10 transition-all duration-300 border border-amber-500/20">
                              <Calendar className="w-5 h-5 text-amber-400 group-hover/stat:scale-110 transition-transform duration-300" />
                            </div>
                            <div>
                              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-0.5">Duration</div>
                              <div className="text-sm font-bold text-white group-hover/stat:text-amber-400 transition-colors duration-300">{itinerary.days?.length} Premium Days</div>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    </div>

                    {/* Concierge Quote - Premium Block */}
                    <div className="relative group/concierge">
                      <div className="absolute -inset-4 bg-gradient-to-r from-primary/15 via-teal/10 to-amber/10 blur-3xl opacity-30 group-hover/concierge:opacity-60 transition-opacity duration-700 rounded-3xl"></div>
                      <div className="glass-card-premium p-12 md:p-14 relative z-10">
                        <div className="absolute top-0 right-0 p-8 opacity-[0.03]">
                          <Award className="w-40 h-40" />
                        </div>
                        <div className="flex items-start gap-8 relative z-10">
                          <div className="hidden lg:block">
                            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary via-teal to-amber p-[2px] shadow-[0_0_30px_rgba(56,189,248,0.2)] group-hover/concierge:shadow-[0_0_50px_rgba(56,189,248,0.3)] transition-shadow duration-500" style={{backgroundSize: '200% 200%'}}>
                              <div className="w-full h-full rounded-full bg-[#0c0e12] flex items-center justify-center">
                                <Award className="w-7 h-7 text-primary" />
                              </div>
                            </div>
                          </div>
                          <div className="flex-1">
                            <div className="text-[10px] font-bold text-primary uppercase tracking-[0.3em] mb-6 flex items-center gap-3">
                              <div className="w-8 h-[1px] bg-gradient-to-r from-primary to-teal"></div>
                              <span className="text-gradient-teal">Executive Director of Concierge</span>
                            </div>
                            <blockquote className="text-2xl md:text-3xl font-light text-slate-200 leading-[1.7] italic serif">
                              "{itinerary.concierge_note}"
                            </blockquote>
                            <div className="mt-8 flex items-center gap-2 text-primary/40 text-[10px] uppercase tracking-[0.2em] font-bold">
                              <div className="w-4 h-[1px] bg-gradient-to-r from-primary to-amber"></div>
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
                                    <div className="w-3 h-3 rounded-full bg-primary shadow-[0_0_20px_#38bdf8] group-hover:shadow-[0_0_30px_#38bdf8] transition-shadow duration-500"></div>
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
                                      className="bg-white/[0.05] hover:bg-gradient-to-r hover:from-primary/20 hover:to-teal/10 p-3 rounded-xl border border-white/[0.06] hover:border-primary/30 text-slate-300 transition-all duration-300 flex items-center gap-2 text-xs font-semibold group/btn"
                                    >
                                      <Navigation className="w-4 h-4 text-primary group-hover/btn:scale-110 transition-transform duration-300" />
                                      Navigate
                                    </a>
                                    <button className="bg-white/[0.05] hover:bg-gradient-to-r hover:from-amber/20 hover:to-rose/10 p-3 rounded-xl border border-white/[0.06] hover:border-amber/30 text-slate-300 transition-all duration-300 flex items-center gap-2 text-xs font-semibold hover:text-amber-400 group/btn">
                                      <Camera className="w-4 h-4 text-amber-400 group-hover/btn:scale-110 transition-transform duration-300" />
                                      Inspiration
                                    </button>
                                  </div>

                                  {act.transport_to_next && (
                                    <div className="glass-card-premium p-8 mt-10 relative overflow-hidden group/trans">
                                      <div className="absolute top-0 right-0 w-40 h-40 bg-primary/[0.04] blur-3xl -mr-10 -mt-10 group-hover/trans:bg-primary/[0.08] transition-all duration-700"></div>
                                      <div className="absolute bottom-0 left-0 w-40 h-40 bg-teal/[0.03] blur-3xl -ml-10 -mb-10 group-hover/trans:bg-teal/[0.06] transition-all duration-700"></div>
                                      <div className="flex flex-col md:flex-row md:items-center justify-between gap-8 relative z-10">
                                        <div className="flex items-center gap-6">
                                          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary/20 via-teal/10 to-primary/5 flex items-center justify-center border border-primary/20 group-hover/trans:border-primary/30 transition-all duration-300 group-hover/trans:shadow-[0_0_25px_rgba(56,189,248,0.15)]">
                                            {act.transport_to_next.mode.toLowerCase().includes('walk') ? <Navigation className="w-7 h-7 text-primary" /> : act.transport_to_next.mode.toLowerCase().includes('train') ? <Train className="w-7 h-7 text-primary" /> : <Bus className="w-7 h-7 text-primary" />}
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
                    {/* Climate Outlook - Premium Widget */}                          <div className="glass-card overflow-hidden group/climate relative">
                      <div className="absolute top-0 right-0 w-40 h-40 bg-amber-400/10 blur-3xl -mr-12 -mt-12 group-hover/climate:bg-amber-400/20 transition-all duration-700"></div>
                      <div className="absolute bottom-0 left-0 w-40 h-40 bg-rose-400/5 blur-3xl -ml-12 -mb-12"></div>
                      <div className="absolute inset-x-0 top-0 h-[1px] bg-gradient-to-r from-transparent via-amber-400/30 via-rose-400/20 to-transparent"></div>
                      <div className="p-8">
                        <h3 className="text-xs font-bold text-white mb-6 flex items-center gap-3 tracking-[0.2em] uppercase">
                          <div className="bg-gradient-to-br from-amber-400/20 to-amber-400/5 p-2 rounded-lg border border-amber-400/20">
                            <Sun className="w-4 h-4 text-amber-400" />
                          </div>
                          <span className="text-gradient-amber">Climate Outlook</span>
                        </h3>
                        {weather ? (
                          <div className="space-y-6 relative z-10">
                            {/* Temperature Display */}
                            <div className="flex items-center gap-5">
                              <div className="text-5xl font-bold text-white tracking-tighter">
                                <span className="bg-gradient-to-br from-amber-200 via-amber-400 to-rose-400 bg-clip-text text-transparent">
                                  {weather.temperature_c?.expected_high != null
                                    ? `${Math.round(weather.temperature_c.expected_high)}°C`
                                    : 'N/A'}
                                </span>
                              </div>
                              <div className="h-12 w-[1px] bg-gradient-to-b from-amber-400/30 to-rose-400/30"></div>
                              <div className="text-[10px] font-bold text-slate-300 uppercase leading-relaxed tracking-wider">Peak Temp<br /><span className="text-gradient-amber">Expected</span></div>
                            </div>
                            {/* Temperature Range */}
                            {weather.temperature_c?.expected_low != null && weather.temperature_c?.expected_high != null && (
                              <div className="flex items-center gap-2 p-2.5 bg-white/[0.03] rounded-lg border border-white/[0.04]">
                                <Thermometer className="w-3.5 h-3.5 text-blue-400" />
                                <span className="text-[11px] font-medium text-blue-200">
                                  {Math.round(weather.temperature_c.expected_low)}°C — {Math.round(weather.temperature_c.expected_high)}°C
                                </span>
                                {weather.temperature_c.typical_range && (
                                  <span className="text-[10px] text-slate-300 ml-auto">{weather.temperature_c.typical_range}</span>
                                )}
                              </div>
                            )}
                            <div className="space-y-3">
                              <div className="flex items-center gap-2.5 p-3.5 bg-white/[0.04] rounded-xl border border-white/[0.06] group-hover/climate:border-amber-400/30 transition-all duration-500 hover:bg-gradient-to-r hover:from-amber-400/10 hover:to-amber-400/5">
                                <div className="w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_15px_#fbbf24] shrink-0 animate-pulse-soft"></div>
                                <span className="text-xs font-semibold text-blue-100 uppercase tracking-wide">{weather.conditions_summary || 'Conditions data pending'}</span>
                              </div>
                              <p className="text-[11px] font-medium text-slate-300 leading-relaxed italic px-1">
                                "{weather.temperature_c?.notes || weather.conditions_summary || "Environmental conditions are optimized for your selected itinerary themes."}"
                              </p>
                            </div>
                            {/* Best Times */}
                            {weather.best_times && weather.best_times.length > 0 && (
                              <div className="space-y-2">
                                <p className="text-[10px] font-bold text-amber-300 uppercase tracking-widest">Best Times</p>
                                <div className="flex flex-wrap gap-1.5">
                                  {weather.best_times.map((t: string, i: number) => (
                                    <span key={i} className="text-[10px] font-medium text-blue-100 bg-amber-400/10 border border-amber-400/15 rounded-full px-2.5 py-1">{t}</span>
                                  ))}
                                </div>
                              </div>
                            )}
                            {/* Activity Suggestions */}
                            {weather.activity_suggestions && weather.activity_suggestions.length > 0 && (
                              <div className="space-y-2">
                                <p className="text-[10px] font-bold text-amber-300 uppercase tracking-widest">Activities</p>
                                <div className="flex flex-wrap gap-1.5">
                                  {weather.activity_suggestions.map((a: string, i: number) => (
                                    <span key={i} className="text-[10px] font-medium text-blue-100 bg-rose-400/10 border border-rose-400/15 rounded-full px-2.5 py-1">{a}</span>
                                  ))}
                                </div>
                              </div>
                            )}
                            {/* Packing */}
                            {weather.packing && weather.packing.length > 0 && (
                              <div className="space-y-2">
                                <p className="text-[10px] font-bold text-amber-300 uppercase tracking-widest">Packing Tips</p>
                                <div className="space-y-1">
                                  {weather.packing.map((p: string, i: number) => (
                                    <div key={i} className="flex items-start gap-2">
                                      <div className="w-1 h-1 rounded-full bg-amber-400 mt-1.5 shrink-0"></div>
                                      <span className="text-[10px] text-slate-200 leading-relaxed">{p}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="py-8 text-center space-y-3">
                            <RefreshCcw className="w-7 h-7 text-amber-400/30 mx-auto animate-spin-slow" />
                            <p className="text-amber-500/50 text-[10px] font-bold uppercase tracking-widest italic">Synching Intelligence...</p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Local Expert - Soul Panel */}
                    <div className="glass-card overflow-hidden group/soul relative">
                      <div className="absolute inset-x-0 bottom-0 h-[1px] bg-gradient-to-r from-transparent via-fuchsia-400/30 via-primary/30 to-transparent"></div>
                      <div className="absolute inset-x-0 top-0 h-[1px] bg-gradient-to-r from-transparent via-primary/30 via-fuchsia-400/20 to-transparent"></div>
                      <div className="absolute top-0 right-0 w-40 h-40 bg-fuchsia-400/5 blur-3xl -mr-8 -mt-8 group-hover/soul:bg-fuchsia-400/10 transition-all duration-700"></div>
                      <div className="absolute bottom-0 left-0 w-40 h-40 bg-primary/5 blur-3xl -ml-8 -mb-8"></div>
                      <div className="p-8">
                        <h3 className="text-xs font-bold text-white mb-6 flex items-center gap-3 tracking-[0.2em] uppercase">
                          <div className="bg-gradient-to-br from-fuchsia-400/20 to-primary/10 p-2 rounded-lg border border-fuchsia-400/20">
                            <Award className="w-4 h-4 text-fuchsia-400" />
                          </div>
                          <span className="bg-gradient-to-r from-fuchsia-400 to-primary bg-clip-text text-transparent">Local Soul Insight</span>
                        </h3>
                        <div className="relative z-10">
                          <div className="text-5xl text-fuchsia-400/20 font-serif absolute -top-5 -left-2 italic leading-none select-none">"</div>
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
                            className="w-full py-3.5 rounded-xl bg-gradient-to-r from-white/[0.04] to-white/[0.02] border border-white/[0.06] text-[11px] font-bold text-slate-400 flex items-center justify-center gap-2 hover:bg-gradient-to-r hover:from-fuchsia-400/15 hover:to-primary/10 hover:text-white hover:border-fuchsia-400/30 transition-all duration-300 uppercase tracking-[0.15em] group/btn cursor-pointer"
                          >
                            EXPAND INTELLIGENCE <ChevronRight className="w-3 h-3 text-primary group-hover/btn:translate-x-0.5 transition-transform duration-300" />
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* Mobility Strategy */}
                    <div className="glass-card overflow-hidden group/mob border-white/5">
                      <div className="absolute inset-x-0 top-0 h-[1px] bg-gradient-to-r from-transparent via-teal-400/30 via-emerald-400/20 to-transparent"></div>
                      <div className="p-8 pb-2">
                        <h3 className="text-xs font-bold text-white mb-6 flex items-center gap-3 tracking-[0.2em] uppercase">
                          <div className="bg-gradient-to-br from-teal-400/20 to-emerald-400/10 p-2 rounded-lg border border-teal-400/20">
                            <Navigation className="w-4 h-4 text-teal-400" />
                          </div>
                          <span className="text-gradient-teal">Mobility Strategy</span>
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
                          <div className="p-5 bg-gradient-to-br from-teal/10 via-emerald/5 to-teal/5 rounded-2xl border border-teal-400/20 hover:border-teal-400/30 hover:from-teal/15 hover:via-emerald/10 transition-all duration-300 group/logic">
                            <div className="flex items-center gap-2 mb-3">
                              <div className="w-5 h-5 rounded-lg bg-gradient-to-br from-teal-400/30 to-emerald-400/20 flex items-center justify-center">
                                <Zap className="w-3 h-3 text-teal-400" />
                              </div>
                              <span className="text-[10px] font-bold text-teal-400/80 uppercase tracking-widest">Coordinated Logic</span>
                            </div>
                            <p className="text-[11px] text-slate-400 leading-relaxed font-light italic">
                              {mobility?.route_optimization?.strategy || "Our agents have calculated the most efficient grouping of destinations to minimize transit fatigue."}
                            </p>
                          </div>
                          <button className="w-full bg-gradient-to-r from-teal/[0.08] via-emerald/[0.05] to-teal/[0.03] hover:from-teal/[0.15] hover:via-emerald/[0.1] hover:to-teal/[0.08] text-teal-400 py-3.5 rounded-xl text-[10px] font-bold tracking-[0.2em] uppercase transition-all duration-300 flex items-center justify-center gap-2.5 border border-teal-400/20 hover:border-teal-400/40 group/btn">
                            <Navigation className="w-4 h-4 group-hover/btn:scale-110 transition-transform duration-300 text-teal-400" />
                            <span className="text-gradient-teal">ACCESS LIVE PLOT</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            ) : null}
          </AnimatePresence>
          )}

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
                  className="relative w-full max-w-4xl max-h-[85vh] overflow-y-auto glass-card-premium p-8 md:p-12 shadow-[0_0_80px_rgba(56,189,248,0.12)] flex flex-col gap-8 custom-scrollbar text-left"
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
                          <span className="w-2 h-2 rounded-full bg-primary shadow-[0_0_10px_#38bdf8]"></span>
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
                              const colorHex = hexMatch ? hexMatch[0] : '#38bdf8';
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
