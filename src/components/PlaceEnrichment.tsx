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

export function PlaceEnrichment({ query }: { query: string }) {
  const [data, setData] = useState<PlaceData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!query) { setLoading(false); return; }
    let cancelled = false;
    setLoading(true);
    axios.get(`${API_BASE_URL}/api/place-enrichment`, { params: { q: query }, timeout: 12000 })
      .then(res => { if (!cancelled) setData(res.data); })
      .catch(() => { if (!cancelled) setData(null); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [query]);

  if (loading) {
    return (
      <div className="animate-pulse bg-white/[0.04] rounded-xl p-4 border border-white/5">
        <div className="h-32 bg-white/5 rounded-lg mb-3" />
        <div className="h-3 bg-white/5 rounded w-1/2" />
      </div>
    );
  }
  if (!data || data.rating == null) return null;

  return (
    <div className="bg-white/[0.04] rounded-xl border border-white/10 overflow-hidden">
      {data.photos && data.photos.length > 0 && (
        <div className="grid grid-cols-2 gap-1">
          {data.photos.slice(0, 2).map((src, i) => (
            <img key={i} src={src} alt={data.place_name} className="h-32 w-full object-cover" loading="lazy"
              onError={(e) => (e.currentTarget.style.display='none')} />
          ))}
        </div>
      )}
      <div className="p-3.5 space-y-2.5">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-1.5 text-amber-400">
            <Star className="w-4 h-4 fill-amber-400" />
            <span className="font-bold text-white text-sm">{data.rating?.toFixed(1)}</span>
            <span className="text-[11px] text-slate-400">({data.user_ratings_total?.toLocaleString()})</span>
          </div>
          <a href={data.google_maps_url} target="_blank" rel="noopener noreferrer" className="text-[10px] font-bold tracking-widest text-primary hover:text-teal-300 flex items-center gap-1">
            <ExternalLink className="w-3 h-3" /> MAPS
          </a>
        </div>
        <div className="text-xs font-semibold text-white line-clamp-1 flex items-center gap-1.5">
          <MapPin className="w-3 h-3 text-primary shrink-0" /> {data.place_name}
        </div>
        {data.reviews && data.reviews.length > 0 && (
          <div className="space-y-2">
            {data.reviews.slice(0, 2).map((r, i) => (
              <div key={i} className="bg-white/[0.03] rounded-lg p-2.5 border border-white/5">
                <div className="flex items-center gap-1.5 mb-1">
                  <span className="text-[11px] font-bold text-white">{r.author_name}</span>
                  <span className="flex">
                    {Array.from({ length: 5 }).map((_, k) => (
                      <Star key={k} className={`w-2.5 h-2.5 ${k < (r.rating || 0) ? 'fill-amber-400 text-amber-400' : 'text-white/20'}`} />
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
          <Camera className="w-3 h-3" /> {data.source === 'google_places' ? 'Google Places' : 'Community'} · <Users className="w-3 h-3" /> {data.user_ratings_total?.toLocaleString()} reviews
        </div>
      </div>
    </div>
  );
}
