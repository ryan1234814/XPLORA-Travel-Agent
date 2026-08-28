import { useState, useEffect, type FormEvent, type KeyboardEvent } from 'react';
import {
  MapPin,
  Zap,
  Search,
  ExternalLink,
  Copy,
  Check,
  ChevronRight,
  Sparkles,
  ArrowRight,
  Navigation,
  Globe,
  MessageSquare,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000');

interface AskPlaceResult {
  place: string;
  question: string;
  conversation_id: string;
  answer_markdown: string;
  location: {
    display_name: string;
    lat: number;
    lng: number;
    address: string;
    type: string;
    google_maps_url: string;
    directions_url: string;
  };
  facts: Array<{ label: string; value: string }>;
  sources: Array<{ title: string; url: string; snippet: string }>;
  followup_suggestions: string[];
}

interface InlineErrorProps {
  message: string;
}

function InlineError({ message }: InlineErrorProps) {
  return (
    <p className="text-[10px] font-medium text-red-400 mt-1.5 ml-1 flex items-center gap-1">
      <span className="text-red-400/60">⚠</span> {message}
    </p>
  );
}

/** Very lightweight markdown-to-HTML for the answer display. Escapes HTML, then applies simple rules. */
function renderMarkdown(md: string): string {
  if (!md) return '';
  // Escape HTML to prevent XSS
  let escaped = md
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Headers
  escaped = escaped.replace(/^### (.+)$/gm, '<h3 class="text-lg font-bold text-white mt-4 mb-2">$1</h3>');
  escaped = escaped.replace(/^## (.+)$/gm, '<h2 class="text-xl font-bold text-white mt-6 mb-3">$1</h2>');
  escaped = escaped.replace(/^# (.+)$/gm, '<h1 class="text-2xl font-bold text-white mt-6 mb-3">$1</h1>');

  // Bold & italic
  escaped = escaped.replace(/\*\*(.+?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>');
  escaped = escaped.replace(/\*(.+?)\*/g, '<em class="text-slate-300">$1</em>');

  // Inline code
  escaped = escaped.replace(/`(.+?)`/g, '<code class="bg-white/10 text-teal-300 px-1.5 py-0.5 rounded text-sm">$1</code>');

  // Unordered list items
  escaped = escaped.replace(/^[•-] (.+)$/gm, '<li class="ml-4 text-slate-300">$1</li>');

  // Horizontal rules
  escaped = escaped.replace(/^---$/gm, '<hr class="border-white/10 my-4" />');

  // Paragraphs: double newlines → paragraph breaks
  escaped = escaped.replace(/\n\n/g, '</p><p class="text-slate-300 leading-relaxed mb-3">');

  // Single newlines → <br/>
  escaped = escaped.replace(/\n/g, '<br/>');

  // Wrap in paragraph
  escaped = `<p class="text-slate-300 leading-relaxed mb-3">${escaped}</p>`;

  // Clean up empty paragraphs
  escaped = escaped.replace(/<p class="text-slate-300 leading-relaxed mb-3"><\/p>/g, '');

  return escaped;
}

export default function AskPlace() {
  const [place, setPlace] = useState('');
  const [question, setQuestion] = useState('');
  const [conversationId, setConversationId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [result, setResult] = useState<AskPlaceResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [followupInput, setFollowupInput] = useState('');
  const [copiedCoords, setCopiedCoords] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  const validate = (): boolean => {
    const errors: Record<string, string> = {};
    if (!place.trim()) {
      errors.place = 'Place is required.';
    } else if (place.trim().length > 120) {
      errors.place = 'Place must be 120 characters or less.';
    }
    if (!question.trim()) {
      errors.question = 'Question is required.';
    } else if (question.trim().length < 3) {
      errors.question = 'Question must be at least 3 characters.';
    } else if (question.trim().length > 500) {
      errors.question = 'Question must be 500 characters or less.';
    }
    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const loadingSteps = [
    'Geocoding location...',
    'Searching the web...',
    'Analyzing results...',
    'Generating answer...',
  ];

  // Advance loading step timer
  useEffect(() => {
    if (!isLoading || loadingStep >= loadingSteps.length - 1) return;
    const stepTimer = setTimeout(() => setLoadingStep(s => Math.min(s + 1, loadingSteps.length - 1)), 2500);
    return () => clearTimeout(stepTimer);
  }, [isLoading, loadingStep, loadingSteps.length]);

  const handleSubmit = async (submitPlace?: string, submitQuestion?: string) => {
    const p = (submitPlace || place).trim();
    const q = (submitQuestion || question).trim();
    if (!p || q.length < 3) return;

    setIsLoading(true);
    setLoadingStep(0);
    setError(null);
    setFieldErrors({});
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/ask-place`, {
        place: p,
        question: q,
        conversation_id: conversationId || undefined,
      });

      const data: AskPlaceResult = response.data;
      setResult(data);
      setConversationId(data.conversation_id);
      setPlace(p);
      setQuestion(q);
    } catch (err: unknown) {
      console.error(err);
      const axiosErr = err as { response?: { data?: { detail?: string } }; code?: string };
      const detail = axiosErr.response?.data?.detail || '';
      if (detail) {
        setError(detail);
      } else if (axiosErr.code === 'ECONNREFUSED' || axiosErr.code === 'ERR_NETWORK') {
        setError('Cannot reach the server. Please check your connection and try again.');
      } else {
        setError('Something went wrong. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleFollowupSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!followupInput.trim() || followupInput.trim().length < 3) return;
    const q = followupInput.trim();
    setFollowupInput('');
    await handleSubmit(place, q);
  };

  const handleFollowupKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleFollowupSubmit(e as unknown as FormEvent);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setQuestion(suggestion);
    handleSubmit(place, suggestion);
  };

  const copyCoords = () => {
    if (result?.location) {
      navigator.clipboard.writeText(`${result.location.lat}, ${result.location.lng}`);
      setCopiedCoords(true);
      setTimeout(() => setCopiedCoords(false), 2000);
    }
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header area with inputs */}
      <div className="shrink-0 px-8 pt-8 pb-6 border-b border-white/5">
        <div className="flex items-center gap-3 mb-6">
          <div className="bg-gradient-to-br from-primary/25 via-teal/15 to-secondary/20 p-2.5 rounded-xl border border-primary/10 shadow-[0_0_20px_rgba(56,189,248,0.15)]">
            <Search className="w-5 h-5 text-primary drop-shadow-[0_0_8px_rgba(56,189,248,0.4)]" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">Ask XPLORA</h2>
            <p className="text-[10px] text-slate-500 italic tracking-wider">Place-Specific Q&amp;A with Web Research</p>
          </div>
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (validate()) handleSubmit();
          }}
          className="space-y-4"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Place input */}
            <div className="space-y-1">
              <div className="relative group input-glow rounded-xl">
                <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-primary transition-colors duration-300" />
                <input
                  type="text"
                  value={place}
                  onChange={(e) => {
                    setPlace(e.target.value);
                    if (fieldErrors.place) setFieldErrors(prev => { const n = { ...prev }; delete n.place; return n; });
                  }}
                  placeholder="e.g. Fushimi Inari, Kyoto"
                  maxLength={120}
                  className={`w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pl-10 pr-4 text-sm focus:border-primary/50 focus:bg-primary/[0.06] focus:shadow-[0_0_20px_rgba(56,189,248,0.06)] transition-all duration-300 outline-none ${fieldErrors.place ? 'border-red-500/60 focus:border-red-500/80' : ''}`}
                />
              </div>
              {fieldErrors.place && <InlineError message={fieldErrors.place} />}
            </div>

            {/* Question input */}
            <div className="space-y-1">
              <div className="relative group input-glow rounded-xl">
                <MessageSquare className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-primary transition-colors duration-300" />
                <input
                  type="text"
                  value={question}
                  onChange={(e) => {
                    setQuestion(e.target.value);
                    if (fieldErrors.question) setFieldErrors(prev => { const n = { ...prev }; delete n.question; return n; });
                  }}
                  placeholder="e.g. Best time to visit? Is it crowded?"
                  maxLength={500}
                  className={`w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pl-10 pr-4 text-sm focus:border-primary/50 focus:bg-primary/[0.06] focus:shadow-[0_0_20px_rgba(56,189,248,0.06)] transition-all duration-300 outline-none ${fieldErrors.question ? 'border-red-500/60 focus:border-red-500/80' : ''}`}
                />
              </div>
              {fieldErrors.question && <InlineError message={fieldErrors.question} />}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={isLoading || !place.trim() || question.trim().length < 3}
              className="text-white font-bold py-3 px-6 rounded-xl shadow-[0_5px_25px_rgba(56,189,248,0.3)] hover:shadow-[0_8px_40px_rgba(56,189,248,0.5)] hover:-translate-y-0.5 active:translate-y-0 transition-all duration-300 disabled:opacity-50 disabled:translate-y-0 flex items-center justify-center gap-2 tracking-[0.12em] text-xs relative overflow-hidden group/btn"
              style={{ background: 'linear-gradient(135deg, #38bdf8 0%, #0284c7 25%, #2dd4bf 65%, #0d9488 100%)' }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/15 to-transparent opacity-0 group-hover/btn:opacity-100 transition-opacity duration-700 -skew-x-12 translate-x-[-100%] group-hover/btn:translate-x-[100%] duration-1000"></div>
              {isLoading ? (
                <>
                  <Search className="w-4 h-4 animate-pulse text-white/70" />
                  RESEARCHING...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  ASK XPLORA
                </>
              )}
            </button>
            {result && (
              <span className="text-[10px] text-slate-500 italic">
                Session: {result.conversation_id.slice(0, 8)}...
              </span>
            )}
          </div>
        </form>
      </div>

      {/* Results area */}
      <div className="flex-1 overflow-y-auto px-8 py-6">
        <AnimatePresence mode="wait">
          {/* Loading state */}
          {isLoading && !result && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center py-20"
            >
              <div className="relative mb-6">
                <motion.div
                  className="w-20 h-20 rounded-full"
                  style={{
                    border: '1.5px solid',
                    borderColor: 'rgba(56,189,248,0.1)',
                    borderTopColor: '#38bdf8',
                  }}
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <Search className="w-6 h-6 text-primary animate-pulse" />
                </div>
              </div>
              <p className="text-slate-400 text-sm">Researching <span className="text-white font-medium">{place}</span>...</p>
              {/* Step progress */}
              <div className="mt-5 flex flex-col items-center gap-2">
                {loadingSteps.map((step, i) => (
                  <div key={i} className={`flex items-center gap-2 text-xs transition-all duration-300 ${
                    i < loadingStep ? 'text-primary opacity-60' :
                    i === loadingStep ? 'text-white' : 'text-slate-600'
                  }`}>
                    <div className={`w-4 h-4 rounded-full border flex items-center justify-center text-[8px] shrink-0 transition-all duration-300 ${
                      i < loadingStep ? 'border-primary/50 bg-primary/20 text-primary' :
                      i === loadingStep ? 'border-primary bg-primary/10 text-primary animate-pulse' :
                      'border-white/10 text-slate-600'
                    }`}>
                      {i < loadingStep ? '✓' : i + 1}
                    </div>
                    <span>{step}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Error state */}
          {error && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-center"
            >
              <p className="text-red-400 text-sm">{error}</p>
            </motion.div>
          )}

          {/* Result */}
          {result && !isLoading && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="grid grid-cols-1 xl:grid-cols-3 gap-6"
            >
              {/* Left 2/3 — Answer */}
              <div className="xl:col-span-2 space-y-6">
                <div className="glass-card-premium p-6">
                  <div
                    className="prose prose-invert max-w-none text-sm leading-relaxed"
                    dangerouslySetInnerHTML={{ __html: renderMarkdown(result.answer_markdown) }}
                  />
                </div>

                {/* Sources */}
                {result.sources.length > 0 && (
                  <div className="glass-card-premium p-5">
                    <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                      <Globe className="w-4 h-4 text-primary" />
                      Sources
                    </h3>
                    <div className="space-y-2">
                      {result.sources.map((src, i) => (
                        <div key={i} className="flex items-start gap-2 text-xs">
                          <span className="text-primary font-mono shrink-0 mt-0.5">[{i + 1}]</span>
                          {src.url ? (
                            <a
                              href={src.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-slate-300 hover:text-primary transition-colors line-clamp-1"
                            >
                              {src.title}
                              <ExternalLink className="w-3 h-3 inline ml-1 opacity-50" />
                            </a>
                          ) : (
                            <span className="text-slate-300 line-clamp-1">{src.title}</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Follow-up suggestions */}
                {result.followup_suggestions.length > 0 && (
                  <div className="space-y-3">
                    <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
                      <Sparkles className="w-3 h-3" />
                      Suggested Follow-ups
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {result.followup_suggestions.map((s, i) => (
                        <button
                          key={i}
                          onClick={() => handleSuggestionClick(s)}
                          className="px-3 py-2 rounded-xl text-xs font-medium bg-white/5 border border-white/10 text-slate-300 hover:bg-primary/10 hover:border-primary/30 hover:text-white transition-all duration-300 flex items-center gap-1.5"
                        >
                          <ChevronRight className="w-3 h-3 text-primary" />
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Follow-up input */}
                <form onSubmit={handleFollowupSubmit} className="flex items-center gap-3 pt-2">
                  <div className="relative flex-1 group input-glow rounded-xl">
                    <input
                      type="text"
                      value={followupInput}
                      onChange={(e) => setFollowupInput(e.target.value)}
                      onKeyDown={handleFollowupKeyDown}
                      placeholder="Ask a follow-up question..."
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-4 pr-4 text-sm focus:border-primary/50 focus:bg-primary/[0.06] transition-all duration-300 outline-none"
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={!followupInput.trim() || followupInput.trim().length < 3}
                    className="bg-gradient-to-r from-primary/80 to-teal/80 text-white font-bold py-3 px-5 rounded-xl hover:shadow-[0_5px_25px_rgba(56,189,248,0.3)] transition-all duration-300 disabled:opacity-50 text-xs flex items-center gap-1.5"
                  >
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </form>
              </div>

              {/* Right 1/3 — Location Card + Facts */}
              <div className="space-y-4">
                {/* Location Card */}
                <div className="glass-card-premium p-5 space-y-4">
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <MapPin className="w-4 h-4 text-primary" />
                    Location Details
                  </h3>

                  <div className="space-y-2">
                    <p className="text-sm text-slate-300 font-medium">{result.location.display_name}</p>
                    <div className="flex items-center gap-2 text-xs text-slate-400">
                      <span className="font-mono bg-white/5 px-2 py-1 rounded">
                        {result.location.lat.toFixed(4)}, {result.location.lng.toFixed(4)}
                      </span>
                      <button
                        onClick={copyCoords}
                        className="text-slate-500 hover:text-primary transition-colors p-1"
                        title="Copy coordinates"
                      >
                        {copiedCoords ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
                      </button>
                    </div>
                    <p className="text-xs text-slate-500">Type: {result.location.type}</p>
                  </div>

                  <div className="flex flex-col gap-2">
                    {result.location.google_maps_url && (
                      <a
                        href={result.location.google_maps_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 px-3 py-2 rounded-xl bg-white/5 border border-white/10 text-xs text-slate-300 hover:bg-primary/10 hover:border-primary/30 hover:text-white transition-all duration-300"
                      >
                        <Globe className="w-3.5 h-3.5 text-primary" />
                        View on Google Maps
                        <ExternalLink className="w-3 h-3 ml-auto opacity-50" />
                      </a>
                    )}
                    {result.location.directions_url && (
                      <a
                        href={result.location.directions_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 px-3 py-2 rounded-xl bg-white/5 border border-white/10 text-xs text-slate-300 hover:bg-primary/10 hover:border-primary/30 hover:text-white transition-all duration-300"
                      >
                        <Navigation className="w-3.5 h-3.5 text-teal" />
                        Get Directions
                        <ExternalLink className="w-3 h-3 ml-auto opacity-50" />
                      </a>
                    )}
                  </div>

                  {/* Mini map embed */}
                  {result.location.lat !== 0 && result.location.lng !== 0 && (
                    <div className="rounded-xl overflow-hidden border border-white/10 aspect-video">
                      <iframe
                        src={`https://www.openstreetmap.org/export/embed.html?bbox=${result.location.lng - 0.01}%2C${result.location.lat - 0.01}%2C${result.location.lng + 0.01}%2C${result.location.lat + 0.01}&layer=mapnik&marker=${result.location.lat}%2C${result.location.lng}`}
                        width="100%"
                        height="100%"
                        style={{ border: 0, minHeight: 180 }}
                        loading="lazy"
                        title="Location Map"
                      />
                    </div>
                  )}
                </div>

                {/* Facts */}
                {result.facts.length > 0 && (
                  <div className="glass-card-premium p-5 space-y-3">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-amber" />
                      Key Facts
                    </h3>
                    <div className="space-y-2">
                      {result.facts.map((fact, i) => (
                        <div key={i} className="flex items-start gap-2 text-xs">
                          <span className="text-primary font-semibold shrink-0">{fact.label}:</span>
                          <span className="text-slate-300">{fact.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {/* Empty state */}
          {!result && !isLoading && !error && (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center py-20 text-center"
            >
              <div className="bg-gradient-to-br from-primary/15 via-primary/5 to-secondary/10 p-8 rounded-3xl border border-primary/10 mb-6">
                <Search className="w-12 h-12 text-primary drop-shadow-[0_0_12px_rgba(56,189,248,0.3)]" />
              </div>
              <h3 className="text-2xl font-bold text-white mb-2">Ask About Any Place</h3>
              <p className="text-slate-400 text-sm max-w-md">
                Enter a place and your question above. XPLORA will research the web and give you
                a comprehensive answer with sources, location details, and helpful facts.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
