import type { SVGProps } from 'react';

/**
 * XPLORA brand mark — a folded paper airplane, the classic symbol of travel.
 * Behaves like a Lucide icon: it inherits `currentColor` (so Tailwind `text-*`
 * and `drop-shadow-*` utilities apply) and is sized via `className`.
 */
export default function BrandMark({ className, ...props }: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      width="24"
      height="24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      {/* Upper wing */}
      <path d="M21 3 3 11l8 1.5L21 3Z" fill="currentColor" />
      {/* Lower folded wing (lighter facet for a dimensional crease) */}
      <path d="M21 3l-10 9.5L13.5 21 21 3Z" fill="currentColor" fillOpacity="0.5" />
    </svg>
  );
}
