import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import {
  Download,
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
  Thermometer,
  Footprints,
  Flame,
  AlertCircle,
  LayoutDashboard,
  LogIn
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import BrandMark from './components/BrandMark';

// Lightweight web-research fallback used when the backend is unreachable or returns no LLM data.
// It uses DuckDuckGo HTML search to build a usable itinerary/weather so the app still works.
async function webSearchFallback(destination: string, duration: number, interests: string[], budget: string, travelDates: string, pace: string, groupSize: number, groupType: string) {
  void budget;
  void travelDates;
  const search = async (query: string) => {
    try {
      const res = await axios.get('https://html.duckduckgo.com/html/', {
        params: { q: query },
        headers: { 'User-Agent': 'XploraTravel/1.0' },
        timeout: 10000,
      });
      const html = res.data as string;
      const links: { title: string; snippet: string; href: string }[] = [];            const sec = typeof document !== 'undefined' ? document.createElement('div') : null;
      if (!sec) return [];
      sec.innerHTML = html;
      sec.querySelectorAll('result__a, result__snippet').forEach((el) => {
        const text = (el.textContent || '').trim();
        if (!text) return;
        if (el.classList.contains('result__a')) {
          const existing = links[links.length - 1];
          if (existing) existing.title = text;
          else links.push({ title: text, snippet: '', href: (el.getAttribute('href') || '').replace(/\/uddg=/, '') });
        } else if (el.classList.contains('result__snippet') && links.length) {
          links[links.length - 1].snippet = text;
        }
      });
      return links.slice(0, 6);
    } catch {
      return [];
    }
  };

  const symbol = destination.toLowerCase().includes('japan') || destination.toLowerCase().includes('tokyo') || destination.toLowerCase().includes('kyoto') || destination.toLowerCase().includes('osaka') ? '¥' : '$';

  const attractions = await search(`${destination} best attractions things to do must see`);
  const food = await search(`${destination} best restaurants local food ${interests.join(' ')}`);
  const tips = await search(`${destination} travel tips local customs etiquette`);

  const activitiesPerDay = { Relaxed: 2, Moderate: 3, Active: 4, Intense: 5 }[pace] || 3;
  const dayThemes = ['Arrival & Discovery', 'Cultural Immersion', 'Local Exploration', 'Hidden Gems', 'Signature Experiences', 'Adventure Day', 'Relaxation & Reflection', 'Farewell & Memories'];

  const days: DayData[] = [];
  for (let d = 1; d <= duration; d++) {
    const activities: ActivityData[] = [];
    activities.push({
      time: '09:00 AM',
      title: `Day ${d} Morning - Discover ${destination}`,
      description: `Start your day exploring the top-rated highlights of ${destination}. ${attractions[0] ? attractions[0].title + ': ' + attractions[0].snippet.slice(0, 120) : 'Immerse yourself in the local culture and scenery.'}`,
      location: destination,
      tag: 'Culture',
      map_query: destination,
      transport_to_next: { mode: 'Walking', duration: '15 min', cost: 'Free', instructions: `Walk to your next stop in ${destination}` }
    });
    const limitCheck = activitiesPerDay;
    if (limitCheck >= 2) {
      activities.push({
        time: '12:30 PM',
        title: `Day ${d} Lunch - Local Flavors`,
        description: `Enjoy authentic local cuisine. ${food[0] ? food[0].title + ': ' + food[0].snippet.slice(0, 120) : 'Try the region\'s most recommended local dishes.'}`,
        location: `${destination} dining area`,
        tag: 'Gastronomy',
        map_query: `restaurants in ${destination}`,
        transport_to_next: { mode: 'Walking', duration: '10 min', cost: 'Free', instructions: `Walk to your afternoon destination` }
      });
    }
    if (limitCheck >= 3) {
      activities.push({
        time: '03:00 PM',
        title: `Day ${d} Afternoon - ${interests[0] || 'Exploration'}`,
        description: `Continue your journey with activities aligned to your interests: ${interests.join(', ')}. ${tips[0] ? 'Local tip: ' + tips[0].snippet.slice(0, 120) : 'Discover hidden gems and local favorites.'}`,
        location: `${destination} attractions`,
        tag: interests[0] || 'Adventure',
        map_query: `things to do in ${destination}`,
      });
    }
    days.push({
      day_number: d,
      day_name: `Day ${d}`,
      theme: dayThemes[(d - 1) % dayThemes.length],
      activities
    });
  }

  return {
    trip_title: `Discover ${destination} - Your ${duration}-Day Journey`,
    overview: `A curated ${duration}-day experience in ${destination} for ${groupSize} ${groupType.toLowerCase()} traveler(s), shaped by your interests in ${interests.join(', ')}.`,
    sustainability_score: 70,
    price_range: `${symbol}0 - ${symbol}9999 (estimate)`,
    concierge_note: `This plan was assembled from live web research on ${new Date().toISOString().slice(0, 10)} when the primary planning service was unavailable. Always confirm opening hours and reservations locally.`,
    days,
    _source: 'web_search_fallback',
    _research: { attractions, food, tips },
  };
}

import AskPlace from './pages/AskPlace';
import SmartPackTracker from './components/SmartPackTracker';
import { PlaceEnrichment } from './components/PlaceEnrichment';
import TripRating from './components/TripRating';
import { generateItineraryICS, generateItineraryPDF } from './utils/export';

// API Configuration
import { API_BASE_URL } from './utils/apiBase';
import { fetchSavedItinerary, refreshAuth, signOut, useAuth, apiError } from './utils/auth';
import { parseAgentField, parseStoredPlan } from './utils/savedTrips';

// Types
export interface ActivityData {
  time: string;
  title: string;
  place_name?: string;
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

export interface DayData {
  day_number: number;
  day_name: string;
  theme: string;
  activities: ActivityData[];
}

export interface ItineraryData {
  trip_title: string;
  overview: string;
  sustainability_score: number;
  price_range: string;
  concierge_note: string;
  days: DayData[];
}

export interface MobilityOption {
  company?: string;
  mode?: string;
  pros_cons?: string;
  why?: string;
}

export interface MobilitySection {
  comparison_tips?: string[];
  options?: MobilityOption[];
}

export interface MobilityData {
  flights?: MobilitySection | string;
  regional_trains_buses?: MobilitySection | string;
  car_rentals?: MobilitySection | string;
  airport_transfers?: MobilitySection | string;
  local_transport?: MobilitySection | string;
  route_optimization?: {
    strategy?: string;
    suggested_area_groupings?: string[];
    sample_day_route_stops?: string[];
    google_maps_directions_url?: string;
  };
}

export interface WeatherData {
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

export interface LocalExpertSection {
  title?: string;
  insights?: string[];
}

export interface LocalExpertData {
  summary?: string;
  contemporary_behaviors?: LocalExpertSection;
  unwritten_customs?: LocalExpertSection;
  folklore_heritage?: LocalExpertSection;
  guidebook_vs_reality?: LocalExpertSection;
  authenticity_signals?: LocalExpertSection;
  sensory_profile?: {
    title?: string;
    sounds?: string[];
    scents?: string[];
    colors?: string[];
  };
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
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
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
  const [localExpert, setLocalExpert] = useState<LocalExpertData | null>(null);
  const [isIntelligenceOpen, setIsIntelligenceOpen] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [expandedMobility, setExpandedMobility] = useState<string | null>(null);
  const navigate = useNavigate();
  const [viewMode, setViewMode] = useState<'itinerary' | 'ask'>('itinerary');

  // Guided planner: three short steps instead of one dense wall of inputs.
  const [step, setStep] = useState(0);
  const [stepError, setStepError] = useState<string | null>(null);
  const [plannerOpen, setPlannerOpen] = useState(true);
  const STEP_COUNT = 3;
  const STEP_FIELDS: string[][] = [
    ['destination', 'groupType', 'groupSize', 'duration'],
    ['interests', 'pace', 'budget', 'dietaryRequirements'],
    ['riskTolerance', 'accommodationPreference', 'languagePreference', 'occasion', 'accessibility'],
  ];
  const STEP_LABELS = ['The journey', 'Your style', 'Finishing touches'];

  /** Go to a step only if every earlier step is already valid. */
  const goToStep = (target: number) => {
    const clamped = Math.max(0, Math.min(STEP_COUNT - 1, target));
    if (clamped <= step) {
      setStep(clamped);
      setStepError(null);
      return;
    }
    const errors = validateFields();
    for (let i = step; i < clamped; i++) {
      const blocking = STEP_FIELDS[i].filter((f) => errors[f]);
      if (blocking.length) {
        setStep(i);
        setStepError(errors[blocking[0]]);
        return;
      }
    }
    setStep(clamped);
    setStepError(null);
  };

  const nextStep = () => {
    const errors = validateFields();
    const blocking = STEP_FIELDS[step].filter((f) => errors[f]);
    if (blocking.length) {
      setStepError(errors[blocking[0]]);
      return;
    }
    setStepError(null);
    setStep((s) => Math.min(STEP_COUNT - 1, s + 1));
  };

  // Account state (shared store backed by the HTTP-only session cookie)
  const { user } = useAuth();
  const [searchParams] = useSearchParams();
  const savedTripIdParam = searchParams.get('trip');
  const destParam = searchParams.get('dest');
  const [savedTripId, setSavedTripId] = useState<number | null>(null);
  const [savedTripTitle, setSavedTripTitle] = useState('');
  const [savedTripBusy, setSavedTripBusy] = useState(false);

  // Landing-page destination cards deep-link here with ?dest=Kyoto — pre-fill the
  // planner so the visitor lands mid-brief instead of on an empty form.
  useEffect(() => {
    if (!destParam) return;
    setDestination(destParam);
    setViewMode('itinerary');
  }, [destParam]);

  // Restore the session on mount. Never blocks the page: /app stays usable
  // signed out, the cookie just decides whether trips get archived.
  useEffect(() => {
    refreshAuth();
  }, []);

  // Reopen a trip chosen on the dashboard: the stored agent_outputs map is fed
  // through the same parsers as a live generation.
  useEffect(() => {
    if (!savedTripIdParam) return;
    const id = Number(savedTripIdParam);
    if (!Number.isInteger(id) || id <= 0 || id === savedTripId) return;
    let cancelled = false;
    (async () => {
      setSavedTripBusy(true);
      setError(null);
      try {
        const stored = await fetchSavedItinerary(id);
        if (cancelled) return;
        const parsed = parseStoredPlan(stored.itinerary_data);
        if (!parsed.itinerary) {
          setError('This saved trip has no itinerary data to display.');
        } else {
          setItinerary(parsed.itinerary);
          setMobility(parsed.mobility);
          setWeather(parsed.weather);
          setLocalExpert(parsed.localExpert as LocalExpertData | null);
        }
        setDestination(stored.destination || '');
        setOrigin(stored.origin || '');
        if (stored.duration) setDuration(stored.duration);
        if (stored.budget) setBudget(stored.budget);
        if (Array.isArray(stored.interests) && stored.interests.length) setSelectedInterests(stored.interests);
        if (stored.travel_dates) setTravelDates(stored.travel_dates);
        if (stored.group_size) setGroupSize(stored.group_size);
        if (stored.group_type) setGroupType(stored.group_type);
        if (stored.pace) setPace(stored.pace);
        if (stored.accommodation_preference) setAccommodationPreference(stored.accommodation_preference);
        if (stored.occasion) setOccasion(stored.occasion);
        if (Array.isArray(stored.dietary_requirements)) setDietaryRequirements(stored.dietary_requirements);
        if (Array.isArray(stored.accessibility)) setAccessibility(stored.accessibility);
        setSavedTripTitle(stored.title || stored.destination || 'Saved trip');
        setSavedTripId(id);
        setActiveTab(0);
        setViewMode('itinerary');
        setPlannerOpen(false);
      } catch (err) {
        if (!cancelled) setError(apiError(err));
      } finally {
        if (!cancelled) setSavedTripBusy(false);
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [savedTripIdParam]);

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

  // Returns the collected errors (empty object when the form is valid) so the
  // wizard can jump back to the step that needs attention.
  const validateFields = (): Record<string, string> => {
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
    return errors;
  };

  const fieldErrorClass = (field: string) => fieldErrors[field]
    ? 'border-red-500/60 focus:border-red-500/80 focus:bg-red-500/[0.06] focus:shadow-[0_0_20px_rgba(239,68,68,0.08)]'
    : '';

  const handleGenerate = async () => {
    const errors = validateFields();
    if (Object.keys(errors).length > 0) {
      // Send the user to the first step that still has a problem.
      const firstBad = STEP_FIELDS.findIndex((fields) => fields.some((f) => errors[f]));
      if (firstBad >= 0) {
        setStep(firstBad);
        const field = STEP_FIELDS[firstBad].find((f) => errors[f]);
        setStepError(field ? errors[field] : null);
      }
      return;
    }
    setStepError(null);

    setIsLoading(true);
    setError(null);
    setFieldErrors({});
    setItinerary(null);

    try {
      let data: Record<string, unknown> | null = null;

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
          risk_tolerance: riskTolerance,
          title: destination ? `Trip to ${destination}` : ''
        }, { timeout: 180000, withCredentials: true });

        data = response.data;
        console.log('[DEBUG] Full API response keys:', Object.keys(data ?? {}));
      } catch (networkErr) {
        console.warn('[WARN] Backend itinerary call failed, falling back to web research:', (networkErr as { message?: string })?.message ?? networkErr);
        data = {
          itinerary_planner: await webSearchFallback(destination, duration, selectedInterests, budget, travelDates, pace, groupSize, groupType),
          weather_analyst: null,
          transport_mobility: null,
          local_expert: null,
        };
      }

      setItinerary(parseAgentField<ItineraryData>(data?.itinerary_planner));
      setMobility(parseAgentField<MobilityData>(data?.transport_mobility));
      setWeather(parseAgentField<WeatherData>(data?.weather_analyst));
      setLocalExpert(parseAgentField<LocalExpertData>(data?.local_expert));

      setActiveTab(0);
      setPlannerOpen(false); // results take the stage; brief stays one click away
    } catch (err) {
      console.error(err);
      const apiErr = err as { response?: { data?: { detail?: string } }; code?: string };
      const detail = apiErr.response?.data?.detail || '';
      if (detail.includes('capacity') || detail.includes('rate limit') || detail.includes('overloaded')) {
        setError("Travel planning is temporarily busy. Please try again in a moment.");
      } else if (detail) {
        setError(detail);
      } else if (apiErr.code === 'ECONNREFUSED' || apiErr.code === 'ERR_NETWORK') {
        setError("Cannot reach the server. Please check your connection and try again.");
      } else {
        setError("Something went wrong while planning your trip. Please try again.");
      }
      return;
    } finally {
      setIsLoading(false);
    }
  };

  // Format an ISO yyyy-mm-dd pick into a readable, timezone-safe string
  const fmtTravelDate = (v: string) => {
    const [y, m, d] = v.split('-').map(Number);
    if (!y || !m || !d) return v;
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return `${months[m - 1]} ${d}, ${y}`;
  };

  // Calendar picks compose the travelDates string the backend already expects
  const handleDatePick = (which: 'start' | 'end', value: string) => {
    const s = which === 'start' ? value : startDate;
    const e = which === 'end' ? value : endDate;
    if (which === 'start') setStartDate(value);
    else setEndDate(value);
    if (!s && !e) setTravelDates('');
    else if (s && e) setTravelDates(`${fmtTravelDate(s)} → ${fmtTravelDate(e)}`);
    else setTravelDates(fmtTravelDate(s || e));
  };

  // Live route plot for the Mobility Strategy panel: prefer the backend's
  // route_optimization output; otherwise build directions from the visible day's
  // activity places. Needs >= 2 distinct stops to form a routable path.
  const livePlot = useMemo(() => {
    const route = mobility?.route_optimization;
    let stops = (route?.sample_day_route_stops || []).filter((s): s is string => typeof s === 'string' && s.trim().length > 0);
    let url = (route?.google_maps_directions_url || '').trim();
    let source: 'backend' | 'itinerary' = 'backend';

    if (stops.length === 0 && itinerary?.days?.[activeTab]?.activities?.length) {
      stops = itinerary.days[activeTab].activities
        .map(a => {
          const place = a.place_name || (a.map_query && !/\b(in|at)\b/i.test(a.map_query) ? a.map_query : '') || a.location;
          if (!place) return '';
          return place.includes(',') ? place : `${place}, ${destination}`;
        })
        .filter(Boolean);
      stops = Array.from(new Set(stops.map(s => s.toLowerCase()))).map(l => stops[stops.findIndex(s => s.toLowerCase() === l)]);
      url = '';
      source = 'itinerary';
    }

    if (stops.length < 2) return null;
    if (!url) {
      const [originStop, destStop, ...midStops] = stops;
      url = `https://www.google.com/maps/dir/?api=1&origin=${encodeURIComponent(originStop)}&destination=${encodeURIComponent(destStop)}` +
        (midStops.length ? `&waypoints=${encodeURIComponent(midStops.join('|'))}` : '');
    }
    return { stops, url, source };
  }, [mobility, itinerary, activeTab, destination]);

  const handleReset = () => {
    setItinerary(null);
    setSavedTripTitle('');
    setSavedTripId(null);
    setMobility(null);
    setWeather(null);
    setLocalExpert(null);
    setDestination('');
    setOrigin('');
    setDuration(3);
    setBudget('Premier');
    setSelectedInterests(['Wellness', 'Gastronomy']);
    setTravelDates('');
    setStartDate('');
    setEndDate('');
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
    setStep(0);
    setStepError(null);
    setPlannerOpen(true);
  };

  const handleResetAndSwitch = (mode: 'itinerary' | 'ask') => {
    if (mode === 'itinerary') {
      handleReset();
    }
    setViewMode(mode);
  };

  const handleExportPDF = () => {
    if (!itinerary) return;
    generateItineraryPDF(itinerary, weather, mobility, destination);
  };

  const handleExportICS = () => {
    if (!itinerary) return;
    generateItineraryICS(itinerary, destination, travelDates);
  };

  return (
    <div className="main-gradient min-h-screen text-slate-200">
      <>
      {/* Top Header Bar: branding + view toggle */}
      <header className="sticky top-0 z-30 border-b border-white/[0.08] bg-black/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-wide items-center justify-between gap-4 px-4 py-2.5 sm:px-6">
          <button
            type="button"
            onClick={() => navigate('/')}
            title="Back to home"
            className="group flex items-center gap-2.5 text-left"
          >
            <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/[0.05] transition-colors duration-300 group-hover:border-sky-500/40">
              <BrandMark className="h-4 w-4 text-sky-300" />
            </span>
            <span className="hidden flex-col sm:flex">
              <span className="text-[15px] font-semibold leading-tight tracking-tight text-white">XPLORA</span>
              <span className="text-[10px] leading-tight text-slate-500">Intelligent Travel Architect</span>
            </span>
          </button>

          {/* View mode toggle */}
          <div className="flex shrink-0 items-center gap-1.5 sm:gap-2">
            {user ? (
              <>
                <Link
                  to="/dashboard"
                  title="Your dashboard"
                  className="hidden items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-[11px] text-slate-300 transition-colors hover:bg-white/[0.08] hover:text-white md:flex"
                >
                  <LayoutDashboard className="h-3.5 w-3.5" />
                  My trips
                </Link>
                <button
                  onClick={() => navigate('/dashboard')}
                  title={user.email}
                  className="max-w-[140px] truncate rounded-full border border-sky-500/30 bg-sky-500/12 px-3 py-1.5 text-[11px] font-medium text-sky-200 transition-colors hover:bg-sky-500/20"
                >
                  {(user.display_name || user.email).split(' ')[0]}
                </button>
                <button
                  onClick={async () => { await signOut(); handleReset(); }}
                  className="rounded-full border border-white/10 px-3 py-1.5 text-[11px] text-slate-400 transition-colors hover:text-white"
                >
                  Sign out
                </button>
              </>
            ) : (
              <button
                onClick={() => navigate('/login')}
                className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-[11px] font-medium text-slate-200 transition-colors hover:bg-white/[0.09] hover:text-white"
              >
                <LogIn className="h-3.5 w-3.5" />
                Sign in
              </button>
            )}

            <div className="flex items-center gap-0.5 rounded-full border border-white/10 bg-white/[0.04] p-0.5">
              <button
                onClick={() => handleResetAndSwitch('itinerary')}
                className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[11px] font-medium transition-all duration-300 ${viewMode === 'itinerary'
                  ? 'bg-white text-black'
                  : 'text-slate-400 hover:text-white'
                }`}
              >
                <BrandMark className="h-3 w-3" />
                Itinerary
              </button>
              <button
                onClick={() => handleResetAndSwitch('ask')}
                className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[11px] font-medium transition-all duration-300 ${viewMode === 'ask'
                  ? 'bg-white text-black'
                  : 'text-slate-400 hover:text-white'
                }`}
              >
                <Search className="h-3 w-3" />
                Ask
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Reopened-from-dashboard / unsaved-session context strip */}
      {viewMode === 'itinerary' && (savedTripBusy || savedTripTitle || (!user && !!itinerary)) && (
        <div className="mx-auto max-w-wide px-4 sm:px-6 pt-4">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2 rounded-xl border border-sky-500/25 bg-sky-500/[0.07] px-4 py-2.5 text-[12px]">
            {savedTripBusy ? (
              <>
                <RefreshCcw className="h-3.5 w-3.5 animate-spin text-sky-300" />
                <span className="text-sky-100">Loading your saved trip…</span>
              </>
            ) : savedTripTitle ? (
              <>
                <LayoutDashboard className="h-3.5 w-3.5 shrink-0 text-sky-300" />
                <span className="truncate font-medium text-sky-100">Viewing saved trip · {savedTripTitle}</span>
                <button
                  onClick={() => { handleReset(); navigate('/app'); }}
                  className="ml-auto text-[11px] font-medium text-sky-300 transition-colors hover:text-white"
                >
                  Plan a new trip
                </button>
              </>
            ) : (
              <>
                <Sparkles className="h-3.5 w-3.5 shrink-0 text-sky-300" />
                <span className="text-slate-200">This trip is not saved to an account yet.</span>
                <Link
                  to="/signup"
                  className="text-[11px] font-medium text-sky-300 transition-colors hover:text-white"
                >
                  Sign up to keep it
                </Link>
                <Link
                  to="/login"
                  className="text-[11px] font-medium text-slate-400 transition-colors hover:text-white"
                >
                  Sign in
                </Link>
              </>
            )}
          </div>
        </div>
      )}

      {/* Trip Planner — guided three-step wizard on black.
          Once a trip exists it collapses to a one-line brief so the results
          own the page; the brief stays one click from being edited. */}
      {viewMode === 'itinerary' && itinerary && !plannerOpen && (
        <section className="mx-auto max-w-wide px-4 pt-5 sm:px-6" aria-label="Trip brief">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-2xl border border-white/[0.09] bg-[#121213] px-4 py-3 sm:px-5">
            <span className="flex min-w-0 items-center gap-2">
              <MapPin className="h-3.5 w-3.5 shrink-0 text-sky-300" aria-hidden="true" />
              <span className="truncate text-[13px] font-medium text-white">{destination.trim() || 'Trip brief'}</span>
            </span>
            <span className="hidden text-[12px] text-slate-500 sm:block">
              {duration} {duration === 1 ? 'day' : 'days'} · {groupSize} {groupSize === 1 ? 'traveller' : 'travellers'} · {pace} · {budget}
            </span>
            <div className="ml-auto flex items-center gap-2">
              <button
                type="button"
                onClick={() => { setPlannerOpen(true); setStep(0); setStepError(null); }}
                className="rounded-full border border-white/[0.10] px-3.5 py-1.5 text-[11.5px] font-medium text-slate-300 transition-colors hover:border-white/25 hover:text-white"
              >
                Edit brief
              </button>
              <button
                type="button"
                onClick={handleGenerate}
                disabled={isLoading}
                className="btn-accent flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-[11.5px] font-medium disabled:opacity-50"
              >
                {isLoading ? <RefreshCcw className="h-3.5 w-3.5 animate-spin" /> : <Zap className="h-3.5 w-3.5" />}
                Regenerate
              </button>
            </div>
          </div>
        </section>
      )}

      {viewMode === 'itinerary' && (!itinerary || plannerOpen) && (
        <section className="mx-auto max-w-wide px-4 pt-5 sm:px-6" aria-label="Trip planner">
          <div className="overflow-hidden rounded-2xl border border-white/[0.09] bg-[#121213]">
            {/* Progress header */}
            <div className="border-b border-white/[0.07] px-5 pt-5 sm:px-7">
              <div className="mb-4 flex items-baseline justify-between gap-4">
                <p className="text-[10px] font-medium uppercase tracking-[0.16em] text-sky-300">
                  Step {step + 1} of {STEP_COUNT}
                </p>
                <button
                  type="button"
                  onClick={handleReset}
                  className="text-[11px] text-slate-500 transition-colors hover:text-slate-300"
                >
                  Reset
                </button>
              </div>

              <ol className="flex gap-2.5" aria-label="Planner progress">
                {STEP_LABELS.map((label, i) => {
                  const state = i === step ? 'current' : i < step ? 'done' : 'todo';
                  return (
                    <li key={label} className="min-w-0 flex-1">
                      <button
                        type="button"
                        onClick={() => goToStep(i)}
                        aria-current={state === 'current' ? 'step' : undefined}
                        aria-label={`Step ${i + 1}: ${label}`}
                        className="group w-full pb-3 text-left"
                      >
                        <span
                          className={`block h-[3px] w-full rounded-full transition-colors duration-300 ${
                            state === 'current'
                              ? 'bg-sky-500'
                              : state === 'done'
                                ? 'bg-sky-500/45'
                                : 'bg-white/[0.10] group-hover:bg-white/20'
                          }`}
                        />
                        <span
                          className={`mt-2 hidden truncate text-[11px] transition-colors duration-300 sm:block ${
                            state === 'current'
                              ? 'font-medium text-white'
                              : state === 'done'
                                ? 'text-slate-400 group-hover:text-slate-200'
                                : 'text-slate-600 group-hover:text-slate-400'
                          }`}
                        >
                          {label}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ol>
            </div>

            {/* Step body */}
            <div className="px-5 py-6 sm:px-7 sm:py-7">
              {stepError && (
                <div
                  role="alert"
                  className="mb-6 flex items-start gap-2.5 rounded-xl border border-red-500/30 bg-red-500/[0.08] px-4 py-3 text-[12.5px] text-red-300"
                >
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{stepError}</span>
                </div>
              )}

              <AnimatePresence mode="wait">
                {/* ---------- Step 1 · The journey ---------- */}
                {step === 0 && (
                  <motion.div
                    key="step-0"
                    initial={{ opacity: 0, x: 14 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -14 }}
                    transition={{ duration: 0.28, ease: [0.28, 0.11, 0.32, 1] }}
                    className="space-y-6"
                  >
                    <div>
                      <h2 className="text-[19px] font-semibold tracking-tight text-white">
                        Where are you headed?
                      </h2>
                      <p className="mt-1 text-[13px] text-slate-400">
                        A city, a region or a whole country — the agents will fill in the details.
                      </p>
                    </div>

                    <div className="grid gap-4 sm:grid-cols-2">
                      <div>
                        <label htmlFor="planner-destination" className="mb-1.5 block text-[11px] font-medium text-slate-400">
                          Destination
                        </label>
                        <div className="relative">
                          <Search aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                          <input
                            id="planner-destination"
                            type="text"
                            value={destination}
                            onChange={(e) => {
                              setDestination(e.target.value);
                              if (fieldErrors.destination) setFieldErrors(prev => { const n = {...prev}; delete n.destination; return n; });
                            }}
                            placeholder="e.g. Kyoto, Japan"
                            aria-invalid={!!fieldErrors.destination}
                            aria-describedby={fieldErrors.destination ? 'err-destination' : undefined}
                            className={`field w-full py-3 pl-9 pr-3 text-[13.5px] font-medium outline-none ${fieldErrorClass('destination')}`}
                          />
                        </div>
                        {fieldErrors.destination && (
                          <p id="err-destination" className="mt-1.5 ml-0.5 flex items-center gap-1 text-[11px] text-red-400">
                            <AlertCircle className="h-3 w-3 shrink-0" /> {fieldErrors.destination}
                          </p>
                        )}
                      </div>

                      <div>
                        <label htmlFor="planner-origin" className="mb-1.5 block text-[11px] font-medium text-slate-400">
                          Leaving from <span className="text-slate-600">(optional)</span>
                        </label>
                        <div className="relative">
                          <MapPin aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                          <input
                            id="planner-origin"
                            type="text"
                            value={origin}
                            onChange={(e) => setOrigin(e.target.value)}
                            placeholder="e.g. New Delhi (DEL)"
                            className="field w-full py-3 pl-9 pr-3 text-[13.5px] outline-none"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="grid gap-6 sm:grid-cols-2">
                      <div>
                        <div className="mb-2 flex items-baseline justify-between">
                          <label htmlFor="planner-duration" className="text-[11px] font-medium text-slate-400">
                            How many days?
                          </label>
                          <span className="text-[13px] font-medium tabular-nums text-sky-300">
                            {duration} {duration === 1 ? 'day' : 'days'}
                          </span>
                        </div>
                        <input
                          id="planner-duration"
                          type="range"
                          min="1"
                          max="14"
                          value={duration}
                          onChange={(e) => {
                            setDuration(parseInt(e.target.value));
                            if (fieldErrors.duration) setFieldErrors(prev => { const n = {...prev}; delete n.duration; return n; });
                          }}
                          aria-valuetext={`${duration} days`}
                          className="w-full cursor-pointer"
                        />
                        <div className="mt-1.5 flex justify-between text-[10px] text-slate-600">
                          <span>1 day</span>
                          <span>2 weeks</span>
                        </div>
                      </div>

                      <div>
                        <div className="mb-2 flex items-baseline justify-between">
                          <label htmlFor="planner-group-size" className="text-[11px] font-medium text-slate-400">
                            Travellers
                          </label>
                          <span className="text-[13px] font-medium tabular-nums text-sky-300">{groupSize}</span>
                        </div>
                        <input
                          id="planner-group-size"
                          type="range"
                          min="1"
                          max="12"
                          value={groupSize}
                          onChange={(e) => {
                            setGroupSize(parseInt(e.target.value));
                            if (fieldErrors.groupSize) setFieldErrors(prev => { const n = {...prev}; delete n.groupSize; return n; });
                          }}
                          aria-valuetext={`${groupSize} travellers`}
                          className="w-full cursor-pointer"
                        />
                        <div className="mt-1.5 flex justify-between text-[10px] text-slate-600">
                          <span>Just me</span>
                          <span>12 people</span>
                        </div>
                      </div>
                    </div>

                    <fieldset>
                      <legend className="mb-1.5 block text-[11px] font-medium text-slate-400">Travelling as</legend>
                      <div className="flex flex-wrap gap-2">
                        {['Solo', 'Couple', 'Family', 'Friends', 'Business'].map((type) => (
                          <button
                            key={type}
                            type="button"
                            aria-pressed={groupType === type}
                            onClick={() => {
                              setGroupType(type);
                              if (fieldErrors.groupType) setFieldErrors(prev => { const n = {...prev}; delete n.groupType; return n; });
                            }}
                            className={`rounded-full border px-3.5 py-1.5 text-[12px] font-medium transition-all duration-300 ${
                              groupType === type
                                ? 'border-sky-500/60 bg-sky-500/15 text-white'
                                : 'border-white/[0.08] bg-white/[0.03] text-slate-400 hover:border-white/20 hover:text-slate-200'
                            }`}
                          >
                            {type}
                          </button>
                        ))}
                      </div>
                    </fieldset>

                    <div>
                      <label htmlFor="planner-dates" className="mb-1.5 block text-[11px] font-medium text-slate-400">
                        When? <span className="text-slate-600">(optional)</span>
                      </label>
                      <input
                        id="planner-dates"
                        type="text"
                        value={travelDates}
                        onChange={(e) => setTravelDates(e.target.value)}
                        placeholder="e.g. Spring 2026, or pick exact dates"
                        className="field w-full px-3.5 py-3 text-[13.5px] outline-none"
                      />
                      <div className="mt-2 grid grid-cols-2 gap-2">
                        <label className="block">
                          <span className="mb-1 block text-[10px] text-slate-500">From</span>
                          <input
                            type="date"
                            value={startDate}
                            max={endDate || undefined}
                            onChange={(e) => handleDatePick('start', e.target.value)}
                            aria-label="Start date of travel"
                            className="field w-full px-2.5 py-2 text-[12px] text-slate-200 outline-none"
                          />
                        </label>
                        <label className="block">
                          <span className="mb-1 block text-[10px] text-slate-500">To</span>
                          <input
                            type="date"
                            value={endDate}
                            min={startDate || undefined}
                            onChange={(e) => handleDatePick('end', e.target.value)}
                            aria-label="End date of travel"
                            className="field w-full px-2.5 py-2 text-[12px] text-slate-200 outline-none"
                          />
                        </label>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* ---------- Step 2 · Your style ---------- */}
                {step === 1 && (
                  <motion.div
                    key="step-1"
                    initial={{ opacity: 0, x: 14 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -14 }}
                    transition={{ duration: 0.28, ease: [0.28, 0.11, 0.32, 1] }}
                    className="space-y-7"
                  >
                    <div>
                      <h2 className="text-[19px] font-semibold tracking-tight text-white">
                        What should the trip feel like?
                      </h2>
                      <p className="mt-1 text-[13px] text-slate-400">
                        Pick what matters. These choices shape the places, the pacing and the price.
                      </p>
                    </div>

                    <fieldset>
                      <legend className="mb-2 text-[11px] font-medium text-slate-400">
                        Interests <span className="text-slate-600">— choose at least one</span>
                      </legend>
                      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
                        {interestsOptions.map((item) => {
                          const on = selectedInterests.includes(item.id);
                          return (
                            <button
                              key={item.id}
                              type="button"
                              aria-pressed={on}
                              onClick={() => toggleInterest(item.id)}
                              className={`flex items-center gap-2 rounded-xl border px-3 py-2.5 text-left text-[12px] font-medium transition-all duration-300 ${
                                on
                                  ? 'border-sky-500/60 bg-sky-500/15 text-white'
                                  : 'border-white/[0.08] bg-white/[0.03] text-slate-400 hover:border-white/20 hover:text-slate-200'
                              }`}
                            >
                              <span className={on ? 'text-sky-300' : 'text-slate-500'}>{item.icon}</span>
                              <span className="truncate">{item.id}</span>
                            </button>
                          );
                        })}
                      </div>
                    </fieldset>

                    <fieldset>
                      <legend className="mb-2 text-[11px] font-medium text-slate-400">Pace</legend>
                      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                        {[
                          { id: 'Relaxed', Icon: Moon, hint: 'Slow mornings' },
                          { id: 'Moderate', Icon: Footprints, hint: 'A steady rhythm' },
                          { id: 'Active', Icon: Zap, hint: 'Full days out' },
                          { id: 'Intense', Icon: Flame, hint: 'Every hour counts' },
                        ].map((p) => {
                          const on = pace === p.id;
                          return (
                            <button
                              key={p.id}
                              type="button"
                              aria-pressed={on}
                              onClick={() => {
                                setPace(p.id);
                                if (fieldErrors.pace) setFieldErrors(prev => { const n = {...prev}; delete n.pace; return n; });
                              }}
                              className={`flex flex-col items-start gap-1 rounded-xl border px-3 py-2.5 transition-all duration-300 ${
                                on
                                  ? 'border-sky-500/60 bg-sky-500/15'
                                  : 'border-white/[0.08] bg-white/[0.03] hover:border-white/20'
                              }`}
                            >
                              <span className="flex items-center gap-1.5">
                                <p.Icon className={`h-3.5 w-3.5 ${on ? 'text-sky-300' : 'text-slate-500'}`} />
                                <span className={`text-[12px] font-medium ${on ? 'text-white' : 'text-slate-300'}`}>{p.id}</span>
                              </span>
                              <span className="text-[10px] text-slate-500">{p.hint}</span>
                            </button>
                          );
                        })}
                      </div>
                    </fieldset>

                    <fieldset>
                      <legend className="mb-2 text-[11px] font-medium text-slate-400">Budget tier</legend>
                      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                        {budgetTiers.map((tier) => {
                          const on = budget === tier;
                          return (
                            <button
                              key={tier}
                              type="button"
                              aria-pressed={on}
                              onClick={() => {
                                setBudget(tier);
                                if (fieldErrors.budget) setFieldErrors(prev => { const n = {...prev}; delete n.budget; return n; });
                              }}
                              className={`rounded-xl border px-3 py-2.5 text-[12px] font-medium transition-all duration-300 ${
                                on
                                  ? 'border-sky-500/60 bg-sky-500/15 text-white'
                                  : 'border-white/[0.08] bg-white/[0.03] text-slate-400 hover:border-white/20 hover:text-slate-200'
                              }`}
                            >
                              {tier}
                            </button>
                          );
                        })}
                      </div>
                    </fieldset>

                    <fieldset>
                      <legend className="mb-2 text-[11px] font-medium text-slate-400">
                        Food needs <span className="text-slate-600">(optional)</span>
                      </legend>
                      <div className="flex flex-wrap gap-2">
                        {['Vegetarian', 'Vegan', 'Halal', 'Kosher', 'Gluten-free', 'Nut allergy', 'Lactose intolerant', 'No restrictions'].map((item) => {
                          const on = dietaryRequirements.includes(item);
                          return (
                            <button
                              key={item}
                              type="button"
                              aria-pressed={on}
                              onClick={() => toggleDietary(item)}
                              className={`rounded-full border px-3 py-1.5 text-[11.5px] font-medium transition-all duration-300 ${
                                on
                                  ? 'border-sky-500/60 bg-sky-500/15 text-white'
                                  : 'border-white/[0.08] bg-white/[0.03] text-slate-400 hover:border-white/20 hover:text-slate-200'
                              }`}
                            >
                              {item}
                            </button>
                          );
                        })}
                      </div>
                    </fieldset>
                  </motion.div>
                )}

                {/* ---------- Step 3 · Finishing touches ---------- */}
                {step === 2 && (
                  <motion.div
                    key="step-2"
                    initial={{ opacity: 0, x: 14 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -14 }}
                    transition={{ duration: 0.28, ease: [0.28, 0.11, 0.32, 1] }}
                    className="space-y-7"
                  >
                    <div>
                      <h2 className="text-[19px] font-semibold tracking-tight text-white">
                        Anything else we should know?
                      </h2>
                      <p className="mt-1 text-[13px] text-slate-400">
                        All optional — skip ahead if you are happy with the defaults.
                      </p>
                    </div>

                    <fieldset>
                      <legend className="mb-2 text-[11px] font-medium text-slate-400">Exploration style</legend>
                      <div className="grid gap-2 sm:grid-cols-3">
                        {[
                          { id: 'Conservative', icon: <Shield aria-hidden="true" className="h-4 w-4" />, label: 'Familiar', desc: 'Well-trodden, safe zones' },
                          { id: 'Balanced', icon: <Globe aria-hidden="true" className="h-4 w-4" />, label: 'Balanced', desc: 'A mix of both' },
                          { id: 'Adventurous', icon: <Sparkles aria-hidden="true" className="h-4 w-4" />, label: 'Off-path', desc: 'The roads less taken' },
                        ].map((r) => {
                          const on = riskTolerance === r.id;
                          return (
                            <button
                              key={r.id}
                              type="button"
                              aria-pressed={on}
                              onClick={() => setRiskTolerance(r.id)}
                              className={`flex items-start gap-2.5 rounded-xl border px-3.5 py-3 text-left transition-all duration-300 ${
                                on
                                  ? 'border-sky-500/60 bg-sky-500/15'
                                  : 'border-white/[0.08] bg-white/[0.03] hover:border-white/20'
                              }`}
                            >
                              <span className={`mt-0.5 ${on ? 'text-sky-300' : 'text-slate-500'}`}>{r.icon}</span>
                              <span className="min-w-0">
                                <span className={`block text-[12.5px] font-medium ${on ? 'text-white' : 'text-slate-300'}`}>{r.label}</span>
                                <span className="block text-[10.5px] text-slate-500">{r.desc}</span>
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </fieldset>

                    <div className="grid gap-4 sm:grid-cols-3">
                      <div>
                        <label htmlFor="planner-accommodation" className="mb-1.5 block text-[11px] font-medium text-slate-400">
                          Stay preference
                        </label>
                        <select
                          id="planner-accommodation"
                          value={accommodationPreference}
                          onChange={(e) => setAccommodationPreference(e.target.value)}
                          className="field w-full cursor-pointer py-3 pl-3 pr-9 text-[13px] outline-none"
                        >
                          {['No preference', 'Hotel', 'Hostel', 'Airbnb/Vacation Rental', 'Boutique/Heritage Stay', 'Camping/Glamping', 'Luxury Resort'].map((opt) => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </select>
                      </div>

                      <div>
                        <label htmlFor="planner-occasion" className="mb-1.5 block text-[11px] font-medium text-slate-400">
                          Occasion
                        </label>
                        <select
                          id="planner-occasion"
                          value={occasion}
                          onChange={(e) => setOccasion(e.target.value)}
                          className="field w-full cursor-pointer py-3 pl-3 pr-9 text-[13px] outline-none"
                        >
                          {['', 'Honeymoon', 'Birthday', 'Anniversary', 'Graduation', 'Proposal', 'Retirement', 'Festival/Celebration'].map((opt) => (
                            <option key={opt} value={opt}>{opt || 'None'}</option>
                          ))}
                        </select>
                      </div>

                      <div>
                        <label htmlFor="planner-language" className="mb-1.5 block text-[11px] font-medium text-slate-400">
                          Language
                        </label>
                        <select
                          id="planner-language"
                          value={languagePreference}
                          onChange={(e) => setLanguagePreference(e.target.value)}
                          className="field w-full cursor-pointer py-3 pl-3 pr-9 text-[13px] outline-none"
                        >
                          {['English only', 'Basic local phrases', 'Conversational local', 'Fluent local'].map((opt) => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <fieldset>
                      <legend className="mb-2 text-[11px] font-medium text-slate-400">
                        Accessibility <span className="text-slate-600">(optional)</span>
                      </legend>
                      <div className="flex flex-wrap gap-2">
                        {['Wheelchair', 'Limited mobility', 'Stroller-friendly', 'Visual support', 'Hearing support', 'None'].map((item) => {
                          const on = accessibility.includes(item);
                          return (
                            <button
                              key={item}
                              type="button"
                              aria-pressed={on}
                              onClick={() => toggleAccessibility(item)}
                              className={`rounded-full border px-3 py-1.5 text-[11.5px] font-medium transition-all duration-300 ${
                                on
                                  ? 'border-sky-500/60 bg-sky-500/15 text-white'
                                  : 'border-white/[0.08] bg-white/[0.03] text-slate-400 hover:border-white/20 hover:text-slate-200'
                              }`}
                            >
                              {item}
                            </button>
                          );
                        })}
                      </div>
                    </fieldset>

                    {/* Recap so the user can confirm at a glance */}
                    <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-4">
                      <p className="mb-2.5 text-[10px] font-medium uppercase tracking-[0.14em] text-slate-500">
                        Your brief
                      </p>
                      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-[12px] sm:grid-cols-4">
                        {[
                          ['Destination', destination.trim() || '—'],
                          ['Length', `${duration} ${duration === 1 ? 'day' : 'days'}`],
                          ['Travellers', `${groupSize} · ${groupType}`],
                          ['Pace', pace],
                          ['Budget', budget],
                          ['Dates', travelDates.trim() || 'Flexible'],
                          ['Stay', accommodationPreference],
                          ['Interests', selectedInterests.length ? `${selectedInterests.length} selected` : '—'],
                        ].map(([k, v]) => (
                          <div key={k} className="min-w-0">
                            <dt className="text-[10px] text-slate-600">{k}</dt>
                            <dd className="truncate font-medium text-slate-200" title={String(v)}>{v}</dd>
                          </div>
                        ))}
                      </dl>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Footer navigation */}
            <div className="flex flex-col-reverse gap-3 border-t border-white/[0.07] bg-white/[0.015] px-5 py-4 sm:flex-row sm:items-center sm:px-7">
              <button
                type="button"
                onClick={() => goToStep(step - 1)}
                disabled={step === 0}
                className="flex items-center justify-center gap-1.5 rounded-full border border-white/[0.09] px-4 py-2.5 text-[12px] font-medium text-slate-300 transition-all duration-300 hover:border-white/20 hover:bg-white/[0.05] hover:text-white disabled:cursor-not-allowed disabled:opacity-30"
              >
                <ArrowRight className="h-3.5 w-3.5 rotate-180" />
                Back
              </button>

              <div className="ml-auto flex w-full gap-2 sm:w-auto">
                {step < STEP_COUNT - 1 ? (
                  <button
                    type="button"
                    onClick={nextStep}
                    className="btn-accent flex flex-1 items-center justify-center gap-2 rounded-full px-6 py-2.5 text-[12.5px] font-medium sm:flex-none"
                  >
                    Continue
                    <ArrowRight className="h-3.5 w-3.5" />
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="btn-accent flex flex-1 items-center justify-center gap-2 rounded-full px-6 py-2.5 text-[12.5px] font-medium disabled:translate-y-0 disabled:opacity-50 sm:flex-none"
                  >
                    {isLoading ? (
                      <>
                        <RefreshCcw className="h-4 w-4 animate-spin" />
                        Designing your voyage…
                      </>
                    ) : (
                      <>
                        <Zap className="h-4 w-4" />
                        Design itinerary
                      </>
                    )}
                  </button>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Main Content Area */}
      <main className="relative mx-auto w-full max-w-wide scroll-smooth px-4 pb-20 sm:px-6">
          {viewMode === 'ask' ? (
            <AskPlace />
          ) : (
          <AnimatePresence mode="wait">
            {!itinerary && !isLoading ? (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                key="welcome"
                className="relative mx-auto flex min-h-[60vh] max-w-2xl flex-col items-center justify-center px-6 py-16 text-center"
              >
                <div className="mb-8 flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.04]">
                  <BrandMark className="h-7 w-7 text-sky-300" />
                </div>

                <h2 className="mb-3 text-[26px] font-semibold tracking-tight text-white sm:text-[32px]">
                  Design your journey above
                </h2>
                <p className="max-w-md text-[14px] leading-relaxed text-slate-400">
                  Fill in the trip form and choose <span className="text-slate-200">Design itinerary</span> — or
                  switch to <span className="text-slate-200">Ask</span> to question the travel brain about any
                  place on Earth.
                </p>
              </motion.div>
            ) : isLoading ? (
              <div key="loading" className="relative flex min-h-[60vh] flex-col items-center justify-center gap-10 py-16">
                <div className="relative flex h-20 w-20 items-center justify-center">
                  <div className="absolute inset-0 animate-spin rounded-full border border-white/10 border-t-sky-500" />
                  <BrandMark className="h-7 w-7 text-sky-300" />
                </div>
                <div className="space-y-3 text-center">
                  <h3 className="text-[20px] font-medium tracking-tight text-white">
                    Crafting your journey
                  </h3>
                  <p className="text-[13px] text-slate-500">
                    Researching routes for {destination || 'your destination'}…
                  </p>
                </div>
              </div>
            ) : itinerary ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                key="itinerary"
                className="py-7"
              >
                <div className="mx-auto grid grid-cols-1 gap-10 xl:grid-cols-4 xl:gap-12">
                  {/* Left Column (3/4) - Main Itinerary */}
                  <div className="space-y-12 xl:col-span-3">
                    {/* Hero Header */}
                    <div className="border-b border-white/[0.08] pb-10">
                      <motion.div
                        initial={{ opacity: 0, y: 16 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.15, duration: 0.6, ease: [0.28, 0.11, 0.32, 1] }}
                      >
                        <div className="mb-5 flex items-center gap-2.5">
                          <BrandMark className="h-3 w-3 text-sky-300" />
                          <span className="text-[11px] text-slate-500">Confirmed itinerary</span>
                        </div>
                        <h2 className="mb-5 text-[clamp(2rem,5vw,3.5rem)] font-semibold leading-[1.05] tracking-[-0.03em] text-white">
                          {itinerary.trip_title}
                        </h2>
                        <p className="mb-9 max-w-3xl text-[16px] leading-relaxed text-slate-400 sm:text-[18px]">
                          {itinerary.overview}
                        </p>
                        <div className="stagger-fade-in flex flex-wrap gap-2.5">
                          <div className="flex items-center gap-3 rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-3">
                            <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/[0.04]">
                              <Tree className="h-4 w-4 text-slate-300" />
                            </div>
                            <div>
                              <div className="text-[10px] text-slate-500">Sustainability</div>
                              <div className="text-[13px] font-medium text-white">Level {itinerary.sustainability_score}%</div>
                            </div>
                          </div>
                          <div className="flex items-center gap-3 rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-3">
                            <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-sky-500/25 bg-sky-500/10">
                              <Zap className="h-4 w-4 text-sky-300" />
                            </div>
                            <div>
                              <div className="text-[10px] text-slate-500">Budget class</div>
                              <div className="text-[13px] font-medium text-white">{itinerary.price_range}</div>
                            </div>
                          </div>
                          <div className="flex items-center gap-3 rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-3">
                            <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-sky-500/25 bg-sky-500/10">
                              <Calendar className="h-4 w-4 text-sky-300" />
                            </div>
                            <div>
                              <div className="text-[10px] text-slate-500">Duration</div>
                              <div className="text-[13px] font-medium text-white">{itinerary.days?.length} days</div>
                            </div>
                          </div>
                        </div>
                        <div className="flex flex-wrap gap-2.5 pt-5">
                          <button
                            onClick={handleExportPDF}
                            className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.05] px-4 py-2 text-[12px] font-medium text-slate-200 transition-colors hover:bg-white/[0.09]"
                          >
                            <Download className="h-3.5 w-3.5" /> Export PDF
                          </button>
                          <button
                            onClick={handleExportICS}
                            className="flex items-center gap-2 rounded-full border border-sky-500/30 bg-sky-500/12 px-4 py-2 text-[12px] font-medium text-sky-200 transition-colors hover:bg-sky-500/20"
                          >
                            <Calendar className="h-3.5 w-3.5" /> Add to calendar
                          </button>
                        </div>
                      </motion.div>
                    </div>

                    {/* Smart Pack + FX Budget Tracker - Traveler Tools */}
                    <SmartPackTracker
                      key={destination || 'destination'}
                      weather={weather}
                      destination={destination}
                      budget={budget}
                      duration={duration}
                      groupSize={groupSize}
                      pace={pace}
                      interests={selectedInterests}
                    />

                    {/* Trip Rating - Social Proof & RLHF Loop */}
                    <TripRating destination={destination} origin={origin} tripTitle={itinerary.trip_title} />

                    {/* Concierge quote — quiet Apple pull-quote on black */}
                    <div className="flat-card p-7 sm:p-9">
                      <div className="flex items-start gap-5">
                        <div className="hidden h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] sm:flex">
                          <Award className="h-5 w-5 text-sky-300" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="mb-4 text-[10px] font-medium uppercase tracking-[0.16em] text-sky-300">
                            Concierge note
                          </div>
                          <blockquote className="text-[17px] leading-[1.7] text-slate-200 sm:text-[20px]">
                            “{itinerary.concierge_note}”
                          </blockquote>
                          <div className="mt-5 text-[11px] text-slate-500">
                            Personal concierge curation
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Day Selection & Itinerary Flow */}
                    <div className="space-y-10">
                      {/* Tabs Bar */}
                      <div className="sticky top-[57px] z-10 -mx-4 flex gap-2 overflow-x-auto px-4 py-4 no-scrollbar" style={{ background: 'linear-gradient(180deg, rgba(0,0,0,0.96) 0%, rgba(0,0,0,0.85) 75%, transparent 100%)' }}>
                        {itinerary.days?.map((day, idx) => (
                          <button
                            key={idx}
                            onClick={() => setActiveTab(idx)}
                            className={`flex min-w-[92px] flex-1 flex-col items-center gap-1 rounded-xl border px-4 py-2.5 transition-all duration-300 tab-pill ${activeTab === idx
                              ? 'tab-pill-active text-white'
                              : 'border-white/[0.07] bg-white/[0.02] text-slate-500 hover:border-white/15 hover:bg-white/[0.05] hover:text-slate-300'
                              }`}
                          >
                            <span className={`text-[10px] uppercase tracking-[0.14em] transition-colors duration-300 ${activeTab === idx ? 'text-sky-300' : 'opacity-60'}`}>Day</span>
                            <span className="text-[15px] font-medium">{day.day_number}</span>
                          </button>
                        ))}
                      </div>

                      {/* Active Day Content */}
                      <AnimatePresence mode="wait">
                        <motion.div
                          key={activeTab}
                          initial={{ opacity: 0, y: 12 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -8 }}
                          transition={{ duration: 0.35, ease: [0.28, 0.11, 0.32, 1] }}
                          className="space-y-10"
                        >
                          <div className="flex items-end justify-between gap-6 border-b border-white/[0.07] pb-8">
                            <div>
                              <div className="mb-3 text-[11px] text-slate-500">Daily focus</div>
                              <h3 className="mb-2 text-[clamp(1.6rem,3.6vw,2.4rem)] font-semibold leading-tight tracking-[-0.025em] text-white">{itinerary.days[activeTab]?.theme}</h3>
                              <p className="text-[14px] text-slate-400">{itinerary.days[activeTab]?.day_name}</p>
                            </div>
                            <div className="hidden md:block">
                              <div className="flex items-center gap-3 rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-3">
                                <div className="text-right">
                                  <div className="text-[10px] text-slate-500">Tempo</div>
                                  <div className="text-[13px] font-medium text-white">Curated fluidity</div>
                                </div>
                                <Activity className="h-4 w-4 text-sky-300" />
                              </div>
                            </div>
                          </div>

                          <div className="relative space-y-10">
                            {/* Timeline connector */}
                            <div className="absolute left-[15px] top-4 bottom-4 w-px bg-gradient-to-b from-sky-500/50 via-white/10 to-transparent" />

                            {itinerary.days[activeTab]?.activities.map((act, idx) => (
                              <motion.div
                                initial={{ opacity: 0, y: 18 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: Math.min(idx * 0.08, 0.3), duration: 0.4, ease: [0.28, 0.11, 0.32, 1] }}
                                key={idx}
                                className="group relative flex gap-5 sm:gap-7"
                              >
                                {/* Timeline Marker */}
                                <div className="flex shrink-0 flex-col items-center">
                                  <div className="relative z-10 flex h-8 w-8 items-center justify-center rounded-full border border-white/15 bg-[#0a0a0b] transition-colors duration-300 group-hover:border-sky-500/60 timeline-dot">
                                    <div className="h-1.5 w-1.5 rounded-full bg-sky-400" />
                                  </div>
                                  <div className="mt-2.5 whitespace-nowrap rounded-lg border border-white/[0.07] bg-white/[0.04] px-2 py-0.5 font-mono text-[10px] text-slate-400">{act.time}</div>
                                </div>

                                <div className="min-w-0 flex-1 space-y-4">
                                  <div className="space-y-2.5">
                                    <div className="flex flex-wrap items-start gap-2.5">
                                      <h4 className="text-[20px] font-medium leading-tight tracking-[-0.02em] text-white transition-colors duration-300 group-hover:text-sky-200 sm:text-[23px]">{act.title}</h4>
                                      <div className="whitespace-nowrap rounded-full border border-white/10 bg-white/[0.05] px-2.5 py-0.5 text-[10px] uppercase tracking-wide text-slate-400">
                                        {act.tag}
                                      </div>
                                    </div>
                                    {/* The actual place the traveller will visit */}
                                    {(act.place_name || act.location) && (
                                      <div className="inline-flex items-center gap-1.5 rounded-lg border border-sky-500/25 bg-sky-500/[0.08] px-2.5 py-1 text-[13px] font-medium text-sky-200">
                                        <MapPin className="h-3.5 w-3.5 shrink-0" />
                                        {act.place_name || act.location}
                                      </div>
                                    )}
                                    {act.location && act.location !== act.place_name && (
                                      <div className="text-[12px] text-slate-500">
                                        {act.location}
                                      </div>
                                    )}
                                  </div>

                                  <p className="max-w-2xl text-[14.5px] leading-relaxed text-slate-400">
                                    {act.description}
                                  </p>

                                  <div className="flex flex-wrap gap-2 pt-0.5">
                                    <a
                                      href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(act.place_name ? `${act.place_name}, ${destination}` : (act.map_query || act.location))}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="flex items-center gap-1.5 rounded-lg border border-white/[0.08] bg-white/[0.04] px-3 py-1.5 text-[12px] text-slate-300 transition-colors hover:border-sky-500/40 hover:bg-sky-500/10 hover:text-sky-200"
                                    >
                                      <Navigation className="h-3.5 w-3.5" />
                                      Navigate
                                    </a>
                                    <button className="flex items-center gap-1.5 rounded-lg border border-white/[0.08] bg-white/[0.04] px-3 py-1.5 text-[12px] text-slate-300 transition-colors hover:border-white/20 hover:text-white">
                                      <Camera className="h-3.5 w-3.5" />
                                      Inspiration
                                    </button>
                                  </div>
                                  {/* Photos + Ratings + Reviews - Social Proof */}
                                  <PlaceEnrichment
                                    query={act.place_name ? `${act.place_name}, ${destination}` : (act.map_query || act.title || act.location)}
                                    offset={activeTab * 3 + idx}
                                  />

                                  {act.transport_to_next && (
                                    <div className="mt-6 rounded-xl border border-white/[0.08] bg-white/[0.025] p-5">
                                      <div className="flex flex-col justify-between gap-5 md:flex-row md:items-center">
                                        <div className="flex items-center gap-4">
                                          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-white/[0.05]">
                                            {act.transport_to_next.mode.toLowerCase().includes('walk') ? <Navigation className="h-5 w-5 text-sky-300" /> : act.transport_to_next.mode.toLowerCase().includes('train') ? <Train className="h-5 w-5 text-sky-300" /> : <Bus className="h-5 w-5 text-sky-300" />}
                                          </div>
                                          <div>
                                            <div className="mb-0.5 text-[10px] uppercase tracking-[0.14em] text-slate-500">Transfer</div>
                                            <div className="flex items-center gap-1.5 text-[15px] font-medium text-white">
                                              {act.transport_to_next.mode}
                                              <ArrowRight className="h-3.5 w-3.5 text-slate-600" />
                                            </div>
                                          </div>
                                        </div>
                                        <div className="flex gap-8 md:gap-10">
                                          <div>
                                            <div className="mb-0.5 text-[10px] text-slate-500">Duration</div>
                                            <div className="flex items-center gap-1.5 text-[13px] font-medium text-white">
                                              <Clock className="h-3.5 w-3.5 text-sky-300" />
                                              {act.transport_to_next.duration}
                                            </div>
                                          </div>
                                          <div>
                                            <div className="mb-0.5 text-[10px] text-slate-500">Cost</div>
                                            <div className="flex items-center gap-1.5 text-[13px] font-medium text-white">
                                              <Wallet className="h-3.5 w-3.5 text-sky-300" />
                                              {act.transport_to_next.cost}
                                            </div>
                                          </div>
                                        </div>
                                      </div>
                                      <div className="mt-4 flex items-start gap-2 border-t border-white/[0.06] pt-4 text-[12px] leading-relaxed text-slate-400">
                                        <span className="text-sky-400">“</span>
                                        <span>{act.transport_to_next.instructions}</span>
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
                  <aside className="space-y-6 xl:col-span-1">
                    {/* Climate Outlook */}
                    <div className="flat-card overflow-hidden">
                      <div className="p-6">
                        <div className="mb-5 flex items-center gap-2.5">
                          <Sun className="h-4 w-4 text-sky-300" />
                          <h3 className="text-[12px] font-medium tracking-wide text-white">Climate outlook</h3>
                        </div>
                        {weather ? (
                          <div className="space-y-5">
                            {/* Temperature Display */}
                            <div className="flex items-end gap-3">
                              <div className="text-[42px] font-semibold leading-none tracking-tight text-white">
                                {weather.temperature_c?.expected_high != null
                                  ? `${Math.round(weather.temperature_c.expected_high)}°`
                                  : 'N/A'}
                              </div>
                              <div className="pb-1 text-[11px] leading-snug text-slate-500">
                                Peak temp<br />expected
                              </div>
                            </div>
                            {/* Temperature Range */}
                            {weather.temperature_c?.expected_low != null && weather.temperature_c?.expected_high != null && (
                              <div className="flex items-center gap-2 rounded-lg border border-white/[0.07] bg-white/[0.03] px-2.5 py-2">
                                <Thermometer className="h-3.5 w-3.5 text-blue-400" />
                                <span className="text-[11px] text-slate-300">
                                  {Math.round(weather.temperature_c.expected_low)}° — {Math.round(weather.temperature_c.expected_high)}°
                                </span>
                                {weather.temperature_c.typical_range && (
                                  <span className="ml-auto text-[10px] text-slate-500">{weather.temperature_c.typical_range}</span>
                                )}
                              </div>
                            )}
                            <div className="space-y-2.5">
                              <div className="flex items-center gap-2 rounded-lg border border-white/[0.07] bg-white/[0.03] px-2.5 py-2">
                                <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-sky-400" />
                                <span className="text-[11px] font-medium text-slate-200">{weather.conditions_summary || 'Conditions data pending'}</span>
                              </div>
                              <p className="px-0.5 text-[11px] leading-relaxed text-slate-500">
                                {weather.temperature_c?.notes || weather.conditions_summary || 'Environmental conditions are optimised for your selected itinerary themes.'}
                              </p>
                            </div>
                            {/* Best Times */}
                            {weather.best_times && weather.best_times.length > 0 && (
                              <div className="space-y-1.5">
                                <p className="text-[10px] font-medium uppercase tracking-[0.12em] text-slate-500">Best times</p>
                                <div className="flex flex-wrap gap-1.5">
                                  {weather.best_times.map((t: string, i: number) => (
                                    <span key={i} className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-0.5 text-[10px] text-slate-300">{t}</span>
                                  ))}
                                </div>
                              </div>
                            )}
                            {/* Activity Suggestions */}
                            {weather.activity_suggestions && weather.activity_suggestions.length > 0 && (
                              <div className="space-y-1.5">
                                <p className="text-[10px] font-medium uppercase tracking-[0.12em] text-slate-500">Activities</p>
                                <div className="flex flex-wrap gap-1.5">
                                  {weather.activity_suggestions.map((a: string, i: number) => (
                                    <span key={i} className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-0.5 text-[10px] text-slate-300">{a}</span>
                                  ))}
                                </div>
                              </div>
                            )}
                            {/* Packing */}
                            {weather.packing && weather.packing.length > 0 && (
                              <div className="space-y-1.5">
                                <p className="text-[10px] font-medium uppercase tracking-[0.12em] text-slate-500">Packing tips</p>
                                <div className="space-y-1">
                                  {weather.packing.map((p: string, i: number) => (
                                    <div key={i} className="flex items-start gap-2">
                                      <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-slate-600" />
                                      <span className="text-[11px] leading-relaxed text-slate-400">{p}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="space-y-2.5 py-6 text-center">
                            <RefreshCcw className="mx-auto h-5 w-5 animate-spin-slow text-sky-300" />
                            <p className="text-[10px] text-slate-500">Synching intelligence…</p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Local Expert */}
                    <div className="flat-card overflow-hidden">
                      <div className="p-6">
                        <div className="mb-4 flex items-center gap-2.5">
                          <Award className="h-4 w-4 text-sky-300" />
                          <h3 className="text-[12px] font-medium tracking-wide text-white">Local soul</h3>
                        </div>
                        <p className="mb-6 text-[12.5px] leading-[1.75] text-slate-400">
                          {localExpert ? (
                            typeof localExpert === 'string'
                              ? (String(localExpert).length > 300 ? String(localExpert).substring(0, 300) + '…' : localExpert)
                              : (localExpert.summary ? (localExpert.summary.length > 300 ? localExpert.summary.substring(0, 300) + '…' : localExpert.summary) : 'Loading…')
                          ) : (
                            'We are gathering contemporary cultural nuances and heritage secrets for this destination to enhance your perspective.'
                          )}
                        </p>
                        <button
                          onClick={() => setIsIntelligenceOpen(true)}
                          className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-white/[0.09] bg-white/[0.04] py-2.5 text-[11px] font-medium text-slate-300 transition-colors duration-300 hover:border-white/20 hover:bg-white/[0.07] hover:text-white"
                        >
                          Read the full brief
                          <ChevronRight className="h-3 w-3" />
                        </button>
                      </div>
                    </div>

                    {/* Mobility Strategy */}
                    <div className="flat-card overflow-hidden">
                      <div className="p-6 pb-2">
                        <div className="mb-4 flex items-center gap-2.5">
                          <Navigation className="h-4 w-4 text-sky-300" />
                          <h3 className="text-[12px] font-medium tracking-wide text-white">Mobility</h3>
                        </div>
                      </div>

                      <div>
                        {[
                          { id: 'flights', label: 'Aerial routes & logistics', icon: <Plane className="w-4 h-4" />, data: mobility?.flights },
                          { id: 'trains', label: 'Regional rail networks', icon: <Train className="w-4 h-4" />, data: mobility?.regional_trains_buses },
                          { id: 'cars', label: 'Private chauffeur & hire', icon: <Car className="w-4 h-4" />, data: mobility?.car_rentals },
                          { id: 'airport', label: 'Protocol transfers', icon: <ShieldCheck className="w-4 h-4" />, data: mobility?.airport_transfers },
                          { id: 'local', label: 'Urban mobility protocol', icon: <Bus className="w-4 h-4" />, data: mobility?.local_transport },
                        ].map((item) => (
                          <div key={item.id} className="group/item border-t border-white/[0.06]">
                            <button
                              onClick={() => setExpandedMobility(expandedMobility === item.id ? null : item.id)}
                              className="flex w-full items-center justify-between px-5 py-3 transition-colors duration-300 hover:bg-white/[0.03]"
                            >
                              <div className="flex items-center gap-3">
                                <div className={`transition-colors duration-300 ${expandedMobility === item.id ? 'text-sky-300' : 'text-slate-500 group-hover/item:text-slate-300'}`}>
                                  {item.icon}
                                </div>
                                <span className={`text-[12px] transition-colors duration-300 ${expandedMobility === item.id ? 'text-white' : 'text-slate-400 group-hover/item:text-slate-200'}`}>
                                  {item.label}
                                </span>
                              </div>
                              <ChevronRight className={`h-3.5 w-3.5 transition-all duration-300 ${expandedMobility === item.id ? 'rotate-90 text-sky-300' : 'text-slate-600 group-hover/item:text-sky-300'}`} />
                            </button>
                            <AnimatePresence>
                              {expandedMobility === item.id && (
                                <motion.div
                                  initial={{ height: 0 }}
                                  animate={{ height: 'auto' }}
                                  exit={{ height: 0 }}
                                  className="overflow-hidden"
                                >
                                  <div className="space-y-3 px-5 pb-4 text-[11px] leading-relaxed text-slate-400">
                                    {item.data ? (
                                      <>
                                        {typeof item.data === 'string' ? (
                                          <div>{item.data}</div>
                                        ) : (
                                          <div className="space-y-3">
                                            {item.data.comparison_tips && (
                                              <div className="space-y-1.5">
                                                <div className="mb-1.5 text-[9px] font-medium uppercase tracking-[0.12em] text-sky-300">Comparison strategy</div>
                                                {item.data.comparison_tips.map((t: string, i: number) => <div key={i} className="flex gap-1.5 text-slate-500"><span className="text-sky-400">•</span> {t}</div>)}
                                              </div>
                                            )}
                                            {item.data.options && (
                                              <div className="space-y-2">
                                                <div className="text-[9px] font-medium uppercase tracking-[0.12em] text-sky-300">Validated providers</div>
                                                {item.data.options.slice(0, 3).map((o: MobilityOption, i: number) => (
                                                  <div key={i} className="rounded-lg border border-white/[0.07] bg-white/[0.03] p-2.5">
                                                    <div className="mb-0.5 text-[11px] font-medium text-white">{o.company || o.mode}</div>
                                                    <div className="text-[10px] leading-relaxed text-slate-500">{o.pros_cons || o.why || 'No additional details'}</div>
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
                                        <Info className="h-3.5 w-3.5" />
                                        <span>Awaiting synchronised data…</span>
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
                      <div className="space-y-4 border-t border-white/[0.06] p-5">
                        <div className="rounded-xl border border-white/[0.07] bg-white/[0.03] p-4">
                          <div className="mb-2 flex items-center gap-2">
                            <Zap className="h-3.5 w-3.5 text-sky-300" />
                            <span className="text-[10px] font-medium uppercase tracking-[0.12em] text-sky-300">Coordinated logic</span>
                          </div>
                          <p className="text-[11px] leading-relaxed text-slate-400">
                            {mobility?.route_optimization?.strategy || 'Our agents have calculated the most efficient grouping of destinations to minimise transit fatigue.'}
                          </p>
                        </div>
                          {livePlot && (
                            <div className="rounded-xl border border-white/[0.07] bg-white/[0.03] p-4">
                              <div className="mb-2 text-[10px] font-medium uppercase tracking-[0.12em] text-sky-300">
                                Route stops{livePlot.source === 'itinerary' ? ' · Day ' + (itinerary?.days[activeTab]?.day_number ?? activeTab + 1) : ''}
                              </div>
                              <ol className="space-y-1.5">
                                {livePlot.stops.map((stop, i) => (
                                  <li key={i} className="flex items-start gap-2 text-[11px] text-slate-400">
                                    <span className="mt-[1px] flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-sky-500/30 bg-sky-500/15 text-[8px] font-medium text-sky-300">
                                      {i + 1}
                                    </span>
                                    <span className="leading-relaxed">{stop}</span>
                                  </li>
                                ))}
                              </ol>
                            </div>
                          )}
                          <button
                            onClick={() => livePlot && window.open(livePlot.url, '_blank', 'noopener,noreferrer')}
                            disabled={!livePlot}
                            title={livePlot ? 'Open the route in Google Maps' : 'No route data available yet'}
                            className={`flex w-full items-center justify-center gap-2 rounded-lg border py-2.5 text-[11px] font-medium transition-all duration-300 ${livePlot
                              ? 'cursor-pointer border-sky-500/30 bg-sky-500/10 text-sky-200 hover:bg-sky-500/20'
                              : 'cursor-not-allowed border-white/[0.06] bg-white/[0.02] text-slate-600'
                            }`}
                          >
                            <Navigation className={`h-3.5 w-3.5 ${livePlot ? 'text-sky-300' : 'text-slate-600'}`} />
                            Open live route
                          </button>
                        </div>
                      </div>
                  </aside>
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
                  className="relative w-full max-w-4xl max-h-[85vh] overflow-y-auto flat-card p-8 md:p-12 flex flex-col gap-8 custom-scrollbar text-left"
                >
                  {/* Header */}
                  {/* Decorative header accent */}
                  <div className="absolute top-0 left-0 right-0 h-[2px] bg-sky-500/40"></div>
                  
                  <div className="flex justify-between items-start gap-4">
                    <div>
                      <span className="text-[10px] font-bold text-sky-300 tracking-[0.3em] uppercase flex items-center gap-2">
                        <BrandMark className="w-3 h-3 text-sky-300" />
                        Deep Intelligence Brief
                      </span>
                      <h2 className="text-[22px] md:text-[28px] font-semibold text-white mt-2 tracking-[-0.025em]">
                        {destination || "Destination"} <span className="accent-label">Living Identity</span>
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
                  <p className="text-slate-300 text-sm md:text-base leading-relaxed italic border-l-2 border-sky-400/40 pl-5 py-2 bg-white/[0.02] rounded-r-xl">
                    "{localExpert?.summary || "Deep heritage insights pending synch."}"
                  </p>

                  {/* Grid of details */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    
                    {/* Contemporary Behaviors */}
                    <div className="p-6 border border-white/[0.08] bg-white/[0.025] rounded-2xl space-y-4 hover:border-sky-400/30 transition-all duration-300 group card-edge">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="w-8 h-8 rounded-lg bg-sky-400/10 flex items-center justify-center group-hover:bg-sky-400/20 transition-colors duration-300">
                          <span className="w-2 h-2 rounded-full bg-sky-400"></span>
                        </div>
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider">{localExpert?.contemporary_behaviors?.title || "Living Rhythms & Trends"}</h3>
                      </div>
                      <ul className="space-y-2.5">
                        {(localExpert?.contemporary_behaviors?.insights || []).map((ins: string, idx: number) => (
                          <li key={idx} className="text-xs font-medium text-slate-400 leading-relaxed flex items-start gap-2.5 bg-white/[0.02] p-2.5 rounded-lg">
                            <span className="text-sky-300 mt-0.5 shrink-0">•</span>
                            <span>{ins}</span>
                          </li>
                        ))}
                        {(!localExpert?.contemporary_behaviors?.insights || localExpert?.contemporary_behaviors?.insights?.length === 0) && (
                          <li className="text-xs font-medium text-slate-500 italic p-2">No contemporary trends indexed.</li>
                        )}
                      </ul>
                    </div>

                    {/* Unwritten Customs */}
                    <div className="p-6 border border-white/[0.08] bg-white/[0.025] rounded-2xl space-y-4 hover:border-sky-400/30 transition-all duration-300 group card-edge">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="w-8 h-8 rounded-lg bg-sky-400/10 flex items-center justify-center group-hover:bg-sky-400/20 transition-colors duration-300">
                          <span className="w-2 h-2 rounded-full bg-sky-400"></span>
                        </div>
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider">{localExpert?.unwritten_customs?.title || "Unwritten Social Codes"}</h3>
                      </div>
                      <ul className="space-y-2.5">
                        {(localExpert?.unwritten_customs?.insights || []).map((ins: string, idx: number) => (
                          <li key={idx} className="text-xs font-medium text-slate-400 leading-relaxed flex items-start gap-2.5 bg-white/[0.02] p-2.5 rounded-lg">
                            <span className="text-sky-300 mt-0.5 shrink-0">•</span>
                            <span>{ins}</span>
                          </li>
                        ))}
                        {(!localExpert?.unwritten_customs?.insights || localExpert?.unwritten_customs?.insights?.length === 0) && (
                          <li className="text-xs font-medium text-slate-500 italic p-2">No social customs indexed.</li>
                        )}
                      </ul>
                    </div>

                    {/* Folklore & Hidden Heritage */}
                    <div className="p-6 border border-white/[0.08] bg-white/[0.025] rounded-2xl space-y-4 hover:border-white/30 transition-all duration-300 group card-edge">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="w-8 h-8 rounded-lg bg-slate-600/10 flex items-center justify-center group-hover:bg-slate-600/20 transition-colors duration-300">
                          <span className="w-2 h-2 rounded-full bg-slate-500"></span>
                        </div>
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider">{localExpert?.folklore_heritage?.title || "Folklore & Hidden Heritage"}</h3>
                      </div>
                      <ul className="space-y-2.5">
                        {(localExpert?.folklore_heritage?.insights || []).map((ins: string, idx: number) => (
                          <li key={idx} className="text-xs font-medium text-slate-400 leading-relaxed flex items-start gap-2.5 bg-white/[0.02] p-2.5 rounded-lg">
                            <span className="text-slate-300 mt-0.5 shrink-0">•</span>
                            <span>{ins}</span>
                          </li>
                        ))}
                        {(!localExpert?.folklore_heritage?.insights || localExpert?.folklore_heritage?.insights?.length === 0) && (
                          <li className="text-xs font-medium text-slate-500 italic p-2">No folklore or heritage stories indexed.</li>
                        )}
                      </ul>
                    </div>

                    {/* Guidebook vs Reality */}
                    <div className="p-6 border border-white/[0.08] bg-white/[0.025] rounded-2xl space-y-4 hover:border-white/30 transition-all duration-300 group card-edge">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="w-8 h-8 rounded-lg bg-slate-700/10 flex items-center justify-center group-hover:bg-slate-700/20 transition-colors duration-300">
                          <span className="w-2 h-2 rounded-full bg-slate-500"></span>
                        </div>
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider">{localExpert?.guidebook_vs_reality?.title || "Guidebook vs. Reality"}</h3>
                      </div>
                      <ul className="space-y-2.5">
                        {(localExpert?.guidebook_vs_reality?.insights || []).map((ins: string, idx: number) => (
                          <li key={idx} className="text-xs font-medium text-slate-400 leading-relaxed flex items-start gap-2.5 bg-white/[0.02] p-2.5 rounded-lg">
                            <span className="text-slate-300 mt-0.5 shrink-0">•</span>
                            <span>{ins}</span>
                          </li>
                        ))}
                        {(!localExpert?.guidebook_vs_reality?.insights || localExpert?.guidebook_vs_reality?.insights?.length === 0) && (
                          <li className="text-xs font-medium text-slate-500 italic p-2">No reality insights indexed.</li>
                        )}
                      </ul>
                    </div>

                    {/* Authenticity Signals */}
                    <div className="p-6 border border-white/[0.08] bg-white/[0.025] rounded-2xl space-y-4 hover:border-white/30 transition-all duration-300 group md:col-span-2 card-edge">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="w-8 h-8 rounded-lg bg-slate-700/10 flex items-center justify-center group-hover:bg-slate-700/20 transition-colors duration-300">
                          <span className="w-2 h-2 rounded-full bg-slate-500"></span>
                        </div>
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider">{localExpert?.authenticity_signals?.title || "Living Authenticity Signals"}</h3>
                      </div>
                      <ul className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                        {(localExpert?.authenticity_signals?.insights || []).map((ins: string, idx: number) => (
                          <li key={idx} className="text-xs font-medium text-slate-400 leading-relaxed flex items-start gap-2.5 bg-white/[0.02] p-2.5 rounded-lg">
                            <span className="text-slate-300 mt-0.5 shrink-0">•</span>
                            <span>{ins}</span>
                          </li>
                        ))}
                        {(!localExpert?.authenticity_signals?.insights || localExpert?.authenticity_signals?.insights?.length === 0) && (
                          <li className="text-xs font-medium text-slate-500 italic md:col-span-2 p-2">No authenticity signals indexed.</li>
                        )}
                      </ul>
                    </div>

                    {/* Sensory Profile */}
                    <div className="p-6 border border-white/[0.08] bg-white/[0.025] rounded-2xl space-y-4 hover:border-white/30 transition-all duration-300 group md:col-span-2 card-edge">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="w-8 h-8 rounded-lg bg-slate-700/10 flex items-center justify-center group-hover:bg-slate-700/20 transition-colors duration-300">
                          <span className="w-2 h-2 rounded-full bg-slate-500"></span>
                        </div>
                        <h3 className="text-xs font-bold text-white uppercase tracking-wider">{localExpert?.sensory_profile?.title || "Sensory Signature"}</h3>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                        
                        {/* Sounds */}
                        <div className="space-y-3">
                          <span className="text-[10px] font-bold text-slate-300 uppercase tracking-widest flex items-center gap-2">
                            <div className="w-3 h-3 rounded bg-slate-700/20 flex items-center justify-center">
                              <span className="w-1 h-1 rounded-full bg-slate-700"></span>
                            </div>
                            Sounds
                          </span>
                          <ul className="space-y-1.5">
                            {(localExpert?.sensory_profile?.sounds || []).map((snd: string, idx: number) => (
                              <li key={idx} className="text-xs font-medium text-slate-400 leading-relaxed flex items-start gap-2 p-2 bg-white/[0.02] rounded-lg">
                                <span className="text-slate-300 mt-0.5">♫</span>
                                <span>{snd}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        {/* Scents */}
                        <div className="space-y-3">
                          <span className="text-[10px] font-bold text-slate-300 uppercase tracking-widest flex items-center gap-2">
                            <div className="w-3 h-3 rounded bg-slate-700/20 flex items-center justify-center">
                              <span className="w-1 h-1 rounded-full bg-slate-700"></span>
                            </div>
                            Scents
                          </span>
                          <ul className="space-y-1.5">
                            {(localExpert?.sensory_profile?.scents || []).map((sct: string, idx: number) => (
                              <li key={idx} className="text-xs font-medium text-slate-400 leading-relaxed flex items-start gap-2 p-2 bg-white/[0.02] rounded-lg">
                                <span className="text-slate-300 mt-0.5">✦</span>
                                <span>{sct}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        {/* Colors */}
                        <div className="space-y-3">
                          <span className="text-[10px] font-bold text-slate-300 uppercase tracking-widest flex items-center gap-2">
                            <div className="w-3 h-3 rounded bg-slate-700/20 flex items-center justify-center">
                              <span className="w-1 h-1 rounded-full bg-slate-700"></span>
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
                <div className="bg-[#1d1d1f] text-white px-5 py-4 rounded-xl flex items-center gap-4 border border-red-500/30 shadow-[0_20px_60px_-15px_rgba(0,0,0,0.8)]">
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
      </>
    </div>
  );
}

export default App;
