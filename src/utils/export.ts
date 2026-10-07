import { jsPDF } from 'jspdf';
import type { ItineraryData, MobilityData, WeatherData } from '../App';

// ---------- Shared helpers ----------

const MARGIN_X = 40;
const CONTENT_WIDTH = 515;
const PAGE_BREAK_Y = 750;

// Brand colors
const DARK = [12, 14, 18] as const;          // #0c0e12
const PRIMARY = [56, 189, 248] as const;     // #38bdf8
const INK = [30, 41, 59] as const;           // dark text on white
const SLATE_BODY = [71, 85, 105] as const;
const SLATE = [100, 116, 139] as const;
const SLATE_LIGHT = [148, 163, 184] as const;

const sanitize = (value: unknown): string => (value == null ? '' : String(value));

// ---------- PDF export ----------

/**
 * Generates a branded A4 itinerary PDF entirely client-side and triggers a download.
 * `mobility` is accepted for API symmetry; the rendered document covers itinerary,
 * badges, concierge note and weather.
 */
export function generateItineraryPDF(
  itinerary: ItineraryData,
  weather: WeatherData | null,
  _mobility: MobilityData | null,
  destination: string
): void {
  const doc = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4' });
  const pageWidth = doc.internal.pageSize.getWidth();
  const days = itinerary.days ?? [];

  let y = 56;

  const drawBrandBar = () => {
    doc.setFillColor(DARK[0], DARK[1], DARK[2]);
    doc.rect(0, 0, pageWidth, 30, 'F');
    doc.setFillColor(PRIMARY[0], PRIMARY[1], PRIMARY[2]);
    doc.rect(0, 30, pageWidth, 2, 'F');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(12);
    doc.setTextColor(PRIMARY[0], PRIMARY[1], PRIMARY[2]);
    doc.text('XPLORA', MARGIN_X, 20);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7);
    doc.setTextColor(SLATE_LIGHT[0], SLATE_LIGHT[1], SLATE_LIGHT[2]);
    doc.text(sanitize(destination).toUpperCase(), pageWidth - MARGIN_X, 20, { align: 'right' });
  };

  const ensureSpace = (height: number) => {
    if (y + height > PAGE_BREAK_Y) {
      doc.addPage();
      y = 40;
      drawBrandBar();
    }
  };

  const textBlock = (
    text: string,
    size: number,
    lineHeight: number,
    color: readonly number[],
    style: 'normal' | 'bold' | 'italic' = 'normal',
    indent = 0
  ) => {
    if (!text.trim()) return;
    const lines = doc.splitTextToSize(text, CONTENT_WIDTH - indent) as string[];
    ensureSpace(lines.length * lineHeight + 4);
    doc.setFont('helvetica', style);
    doc.setFontSize(size);
    doc.setTextColor(color[0], color[1], color[2]);
    doc.text(lines, MARGIN_X + indent, y);
    y += lines.length * lineHeight + 4;
  };

  drawBrandBar();

  // ----- Page 1: title, overview, badges, concierge note -----
  textBlock(itinerary.trip_title || 'Your Itinerary', 22, 28, INK, 'bold');
  textBlock(itinerary.overview || '', 10, 14, SLATE_BODY);

  const badges = [
    {
      label: 'SUSTAINABILITY',
      value: `Level ${itinerary.sustainability_score ?? 0}%`,
      fill: [236, 253, 245] as const,
      border: [167, 243, 208] as const,
      color: [6, 95, 70] as const,
    },
    {
      label: 'BUDGET CLASS',
      value: sanitize(itinerary.price_range),
      fill: [239, 246, 255] as const,
      border: [191, 219, 254] as const,
      color: [29, 78, 216] as const,
    },
    {
      label: 'DURATION',
      value: `${days.length} Days`,
      fill: [255, 251, 235] as const,
      border: [254, 215, 170] as const,
      color: [146, 64, 14] as const,
    },
  ];

  const pillHeight = 44;
  ensureSpace(pillHeight);
  let badgeX = MARGIN_X;
  for (const badge of badges) {
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(7);
    const labelWidth = doc.getTextWidth(badge.label);
    doc.setFontSize(10);
    const valueWidth = doc.getTextWidth(badge.value);
    const pillWidth = Math.max(labelWidth, valueWidth) + 24;
    doc.setFillColor(badge.fill[0], badge.fill[1], badge.fill[2]);
    doc.setDrawColor(badge.border[0], badge.border[1], badge.border[2]);
    doc.roundedRect(badgeX, y, pillWidth, pillHeight, 8, 8, 'FD');
    doc.setFontSize(7);
    doc.setTextColor(badge.color[0], badge.color[1], badge.color[2]);
    doc.text(badge.label, badgeX + 12, y + 16);
    doc.setFontSize(10);
    doc.setTextColor(badge.color[0], badge.color[1], badge.color[2]);
    doc.text(badge.value, badgeX + 12, y + 32);
    badgeX += pillWidth + 10;
  }
  y += pillHeight + 14;

  if (sanitize(itinerary.concierge_note).trim()) {
    const note = `"${itinerary.concierge_note}"`;
    const lines = doc.splitTextToSize(note, CONTENT_WIDTH - 24) as string[];
    const boxHeight = lines.length * 13 + 24;
    ensureSpace(boxHeight);
    doc.setFillColor(248, 250, 252);
    doc.setDrawColor(226, 232, 240);
    doc.roundedRect(MARGIN_X, y, CONTENT_WIDTH, boxHeight, 8, 8, 'FD');
    doc.setDrawColor(PRIMARY[0], PRIMARY[1], PRIMARY[2]);
    doc.setLineWidth(1.5);
    doc.line(MARGIN_X, y, MARGIN_X, y + boxHeight);
    doc.setFont('helvetica', 'italic');
    doc.setFontSize(9);
    doc.setTextColor(SLATE_BODY[0], SLATE_BODY[1], SLATE_BODY[2]);
    doc.text(lines, MARGIN_X + 12, y + 18);
    y += boxHeight + 16;
  }

  // ----- Itinerary days -----
  for (const day of days) {
    const dayHeader = `DAY ${day.day_number} - ${sanitize(day.theme)}`.toUpperCase();
    ensureSpace(40);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(DARK[0], DARK[1], DARK[2]);
    doc.text(dayHeader, MARGIN_X, y);
    const headerWidth = doc.getTextWidth(dayHeader);
    doc.setDrawColor(PRIMARY[0], PRIMARY[1], PRIMARY[2]);
    doc.setLineWidth(1.5);
    doc.line(MARGIN_X, y + 4, MARGIN_X + headerWidth, y + 4);
    y += 20;

    for (const activity of day.activities ?? []) {
      textBlock(`${sanitize(activity.time)}  |  ${sanitize(activity.tag)}`, 9, 13, SLATE_LIGHT, 'normal', 2);
      textBlock(activity.title || 'Activity', 11, 16, INK, 'bold', 2);
      textBlock(activity.description || '', 9, 13, SLATE_BODY, 'normal', 2);
      if (activity.location) {
        textBlock(`📍 ${activity.location}`, 9, 13, SLATE, 'normal', 2);
      }
      if (activity.transport_to_next) {
        const t = activity.transport_to_next;
        const details = [t.mode, t.duration, t.cost].filter(Boolean).join(' · ');
        const instructions = sanitize(t.instructions);
        textBlock(`→ ${details}${instructions ? ` - ${instructions}` : ''}`, 9, 13, SLATE, 'normal', 2);
      }
      y += 8;
    }
    y += 10;
  }

  // ----- Weather box on the last page -----
  if (
    weather &&
    (sanitize(weather.conditions_summary) ||
      weather.temperature_c?.typical_range ||
      weather.temperature_c?.expected_low != null ||
      weather.temperature_c?.expected_high != null ||
      (weather.packing ?? []).length > 0)
  ) {
    const rows: string[] = [];
    const conditions = sanitize(weather.conditions_summary);
    if (conditions) rows.push(conditions);
    if (weather.temperature_c?.typical_range) {
      rows.push(`Typical temperature: ${weather.temperature_c.typical_range}`);
    } else if (weather.temperature_c?.expected_low != null || weather.temperature_c?.expected_high != null) {
      const low = weather.temperature_c.expected_low;
      const high = weather.temperature_c.expected_high;
      rows.push(`Expected temperature: ${low != null ? `${low}°C` : '—'} to ${high != null ? `${high}°C` : '—'}`);
    }
    if ((weather.packing ?? []).length > 0) {
      rows.push('Packing essentials:');
      for (const item of weather.packing ?? []) rows.push(`• ${sanitize(item)}`);
    }

    const wrappedRows = rows.map((row) => doc.splitTextToSize(row, CONTENT_WIDTH - 24) as string[]);
    const boxHeight = 34 + wrappedRows.reduce((count, rowLines) => count + rowLines.length * 13, 0);
    ensureSpace(boxHeight);
    doc.setFillColor(248, 250, 252);
    doc.setDrawColor(226, 232, 240);
    doc.roundedRect(MARGIN_X, y, CONTENT_WIDTH, boxHeight, 8, 8, 'FD');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(10);
    doc.setTextColor(DARK[0], DARK[1], DARK[2]);
    doc.text('WEATHER & PACKING', MARGIN_X + 12, y + 18);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9);
    doc.setTextColor(SLATE_BODY[0], SLATE_BODY[1], SLATE_BODY[2]);
    let textY = y + 34;
    for (const rowLines of wrappedRows) {
      doc.text(rowLines, MARGIN_X + 12, textY);
      textY += rowLines.length * 13;
    }
    y += boxHeight + 16;
  }

  // ----- Footer: page x/y -----
  const totalPages = doc.getNumberOfPages();
  for (let page = 1; page <= totalPages; page++) {
    doc.setPage(page);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    doc.setTextColor(SLATE_LIGHT[0], SLATE_LIGHT[1], SLATE_LIGHT[2]);
    doc.text(`${page} / ${totalPages}`, pageWidth - MARGIN_X, 826, { align: 'right' });
  }

  doc.save(`XPLORA-${sanitize(destination).replace(/\s+/g, '_')}-${days.length}Days.pdf`);
}

// ---------- ICS export ----------

const ICS_LINE_BREAK = '\r\n';

const escapeICS = (value: string): string =>
  value
    .replace(/\\/g, '\\\\')
    .replace(/;/g, '\\;')
    .replace(/,/g, '\\,')
    .replace(/\r?\n/g, '\\n');

/** Formats a Date as YYYYMMDDTHHmmSSZ (UTC). */
const formatICSDate = (date: Date): string => date.toISOString().replace(/[-:]/g, '').split('.')[0] + 'Z';

/**
 * Parses the travel dates input into a start date.
 * Handles native `Date` strings plus the calendar format
 * "Dec 15, 2026 → Dec 22, 2026" (the start date wins); falls back to today.
 */
const parseTravelDates = (travelDates: string): Date => {
  if (travelDates) {
    const parsed = new Date(travelDates);
    if (!isNaN(parsed.getTime())) return parsed;

    const months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];
    const match = travelDates.match(/^([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})/);
    if (match) {
      const month = months.indexOf(match[1].slice(0, 3).toLowerCase());
      const day = parseInt(match[2], 10);
      const year = parseInt(match[3], 10);
      if (month !== -1 && day >= 1 && day <= 31) {
        // UTC keeps the exported day exact regardless of the viewer's timezone
        return new Date(Date.UTC(year, month, day));
      }
    }
  }
  return new Date();
};

/** Parses "09:00 AM" / "3:00 PM" style times onto the base date; defaults to 09:00. */
const parseTimeToDate = (base: Date, timeStr: string): Date => {
  const match = (timeStr || '').match(/(\d{1,2}):(\d{2})\s*(AM|PM)?/i);
  const date = new Date(base);
  if (!match) {
    date.setHours(9, 0, 0, 0);
    return date;
  }
  let hours = parseInt(match[1], 10);
  const minutes = parseInt(match[2], 10);
  const meridiem = (match[3] || '').toUpperCase();
  if (meridiem === 'PM' && hours < 12) hours += 12;
  if (meridiem === 'AM' && hours === 12) hours = 0;
  date.setHours(hours, minutes, 0, 0);
  return date;
};

/**
 * Builds a hand-rolled RFC 5545 VCALENDAR string (no dependencies) with one
 * VEVENT per activity and triggers a .ics download.
 */
export function generateItineraryICS(itinerary: ItineraryData, destination: string, travelDates: string): void {
  const startDate = parseTravelDates(travelDates);
  const days = itinerary.days ?? [];
  const stamp = formatICSDate(new Date());

  let ics = 'BEGIN:VCALENDAR' + ICS_LINE_BREAK;
  ics += 'VERSION:2.0' + ICS_LINE_BREAK;
  ics += 'PRODID:-//XPLORA//Itinerary//EN' + ICS_LINE_BREAK;
  ics += 'CALSCALE:GREGORIAN' + ICS_LINE_BREAK;
  ics += 'METHOD:PUBLISH' + ICS_LINE_BREAK;

  days.forEach((day, dayIndex) => {
    const dayOffset = (day.day_number ?? dayIndex + 1) - 1;
    const dayDate = new Date(startDate);
    dayDate.setDate(startDate.getDate() + dayOffset);

    (day.activities ?? []).forEach((activity, activityIndex) => {
      const dtstart = parseTimeToDate(dayDate, activity.time);
      const dtend = new Date(dtstart.getTime() + 1.5 * 60 * 60 * 1000); // 1.5h default
      const summary = escapeICS(activity.title || 'Activity');
      const description = escapeICS([activity.description, activity.location].filter(Boolean).join(' - '));
      const location = escapeICS(activity.location || '');

      ics += 'BEGIN:VEVENT' + ICS_LINE_BREAK;
      ics += `UID:${day.day_number ?? dayIndex + 1}-${activityIndex}@xplora.local` + ICS_LINE_BREAK;
      ics += `DTSTAMP:${stamp}` + ICS_LINE_BREAK;
      ics += `DTSTART:${formatICSDate(dtstart)}` + ICS_LINE_BREAK;
      ics += `DTEND:${formatICSDate(dtend)}` + ICS_LINE_BREAK;
      ics += `SUMMARY:${summary}` + ICS_LINE_BREAK;
      ics += `DESCRIPTION:${description}` + ICS_LINE_BREAK;
      if (location) ics += `LOCATION:${location}` + ICS_LINE_BREAK;
      ics += 'END:VEVENT' + ICS_LINE_BREAK;
    });
  });

  ics += 'END:VCALENDAR' + ICS_LINE_BREAK;

  const filename = `XPLORA-${sanitize(destination)}-${days.length}Days.ics`;
  const blob = new Blob([ics], { type: 'text/calendar;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  setTimeout(() => URL.revokeObjectURL(url), 100);
}