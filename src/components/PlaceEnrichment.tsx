import { useEffect, useState } from 'react';
import axios from 'axios';
import { Star, MapPin, Camera, Users, ExternalLink } from 'lucide-react';

import { API_BASE_URL } from '../utils/apiBase';

interface PlaceData {
  query: string;
  place_name: string;
  rating: number | null;
  user_ratings_total: number;
  photos: string[];
  reviews: { author_name: string; rating: number; text: string; time: string; profile_photo?: string }[];
  google_maps_url: string;
  source: string;
}

const ATTRIBUTION: Record<string, string> = {
  google_places: 'Google Places',
  wikimedia: 'Wikimedia Commons',
};

export function PlaceEnrichment({ query, offset = 0 }: { query: string; offset?: number }) {
  const [data, setData] = useState<PlaceData | null>(null);
  // The place+offset the loaded data belongs to. Loading is derived from it, so a new
  // activity never needs a synchronous setState inside the effect.
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const requestKey = `${offset}::${query}`;

  useEffect(() => {
    if (!query) return;
    let cancelled = false;
    axios.get(`${API_BASE_URL}/api/place-enrichment`, { params: { q: query, offset }, timeout: 12000 })
      .then(res => { if (!cancelled) { setData(res.data); setLoadedKey(requestKey); } })
      // Unresolvable place: fall through to the guard below, which renders nothing
      // rather than invented imagery.
      .catch(() => { if (!cancelled) { setData(null); setLoadedKey(requestKey); } });
    return () => { cancelled = true; };
  }, [query, offset, requestKey]);

  if (!query) return null;

  if (loadedKey !== requestKey) {
    return (
      <div className="animate-pulse bg-white/[0.04] rounded-xl p-4 border border-white/5">
        <div className="h-32 bg-white/5 rounded-lg mb-3" />
        <div className="h-3 bg-white/5 rounded w-1/2" />
      </div>
    );
  }

  const photos = data?.photos ?? [];
  const reviews = data?.reviews ?? [];
  const hasRating = data != null && data.rating != null;
  // Nothing real to show — better an empty gap than invented imagery.
  if (!data || (photos.length === 0 && !hasRating && reviews.length === 0)) return null;

  const placeName = data.place_name || query;

  return (
    <div className="bg-white/[0.04] rounded-xl border border-white/10 overflow-hidden">
      {photos.length > 0 && (
        <div className={`grid gap-1 ${photos.length > 1 ? 'grid-cols-2' : 'grid-cols-1'}`}>
          {photos.slice(0, 2).map((src, i) => (
            <img key={`${src}-${i}`} src={src} alt={placeName} className="h-32 w-full object-cover" loading="lazy"
              onError={(e) => (e.currentTarget.style.display='none')} />
          ))}
        </div>
      )}
      <div className="p-3.5 space-y-2.5">
        {/* The actual place being visited */}
        <div className="text-sm font-bold text-white flex items-start gap-1.5">
          <MapPin className="w-3.5 h-3.5 text-sky-300 shrink-0 mt-0.5" />
          <span className="line-clamp-2">{placeName}</span>
        </div>
        <div className="flex items-center justify-between gap-2">
          {hasRating ? (
            <div className="flex items-center gap-1.5 text-sky-300">
              <Star className="w-4 h-4 fill-sky-400" />
              <span className="font-bold text-white text-sm">{data.rating?.toFixed(1)}</span>
              <span className="text-[11px] text-slate-400">({data.user_ratings_total?.toLocaleString()})</span>
            </div>
          ) : (
            <span className="text-[11px] text-slate-500">Rating not verified</span>
          )}
          <a href={data.google_maps_url} target="_blank" rel="noopener noreferrer" className="text-[10px] font-bold tracking-widest text-sky-300 hover:text-sky-300 flex items-center gap-1">
            <ExternalLink className="w-3 h-3" /> MAPS
          </a>
        </div>
        {reviews.length > 0 && (
          <div className="space-y-2">
            {reviews.slice(0, 2).map((r, i) => (
              <div key={i} className="bg-white/[0.03] rounded-lg p-2.5 border border-white/5">
                <div className="flex items-center gap-1.5 mb-1">
                  <span className="text-[11px] font-bold text-white">{r.author_name}</span>
                  <span className="flex">
                    {Array.from({ length: 5 }).map((_, k) => (
                      <Star key={k} className={`w-2.5 h-2.5 ${k < (r.rating || 0) ? 'fill-sky-400 text-sky-300' : 'text-white/20'}`} />
                    ))}
                  </span>
                  <span className="text-[10px] text-slate-500 ml-auto">{r.time}</span>
                </div>
                <p className="text-[11px] text-slate-300 leading-relaxed line-clamp-3">"{r.text}"</p>
              </div>
            ))}
          </div>
        )}
        <div className="flex items-center gap-2 text-[10px] text-slate-500">
          <Camera className="w-3 h-3" /> {ATTRIBUTION[data.source] || 'Community'}
          {hasRating && (
            <>
              · <Users className="w-3 h-3" /> {data.user_ratings_total?.toLocaleString()} reviews
            </>
          )}
        </div>
      </div>
    </div>
  );
}
