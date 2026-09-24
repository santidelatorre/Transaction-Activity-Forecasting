// Analysis layer: validated data -> descriptive historical context.
// Everything here is a plain description of OBSERVED transactions.
// It never predicts, never scores, never infers a merchant family from a
// description, and never combines amounts across currencies.

import { ALLOWED_LABELS, NONE_LABEL } from './constants';

export function formatDate(iso) {
    if (!iso) return '\u2014';
    return String(iso).slice(0, 10);
}

export function formatDateTime(iso) {
    if (!iso) return '\u2014';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return `${d.toISOString().slice(0, 10)} ${d.toISOString().slice(11, 16)}`;
}

export function formatAmount(value, currency) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '\u2014';
    return `${n.toFixed(2)} ${currency ?? ''}`.trim();
}

function daysBetween(aIso, bIso) {
    const a = new Date(aIso).getTime();
    const b = new Date(bIso).getTime();
    return Math.round(Math.abs(b - a) / 86400000);
}

/**
 * Group transactions by client id. Repeated ids are expected.
 */
export function groupByClient(rows) {
    const map = new Map();
    (Array.isArray(rows) ? rows : []).forEach((row) => {
        const list = map.get(row.client_id);
        if (list) list.push(row);
        else map.set(row.client_id, [row]);
    });
    map.forEach((list) => list.sort((a, b) => a.timestamp.localeCompare(b.timestamp)));
    return map;
}

/**
 * Build the overview table rows: one entry per client in SAMPLE ORDER.
 */
export function buildClientRows(order, predictions, byClient) {
    return (Array.isArray(order) ? order : []).map((id) => {
        const history = byClient.get(id) ?? [];
        const last = history.length > 0 ? history[history.length - 1].timestamp : null;

        return {
            client_id: id,
            prediction: predictions.get(id) ?? null,
            transaction_count: history.length,
            last_observed: last,
        };
    });
}

/**
 * Distribution of PREDICTIONS across the eight labels.
 * This describes what was predicted \u2014 it says nothing about accuracy.
 */
export function buildDistribution(clientRows) {
    const counts = {};
    ALLOWED_LABELS.forEach((label) => {
        counts[label] = 0;
    });

    (Array.isArray(clientRows) ? clientRows : []).forEach((row) => {
        if (row.prediction && Object.prototype.hasOwnProperty.call(counts, row.prediction)) {
            counts[row.prediction] += 1;
        }
    });

    const total = Object.values(counts).reduce((sum, n) => sum + n, 0);

    return ALLOWED_LABELS.map((label) => ({
        label,
        count: counts[label],
        share: total > 0 ? counts[label] / total : 0,
    }));
}

export function buildSummary(clientRows) {
    const rows = Array.isArray(clientRows) ? clientRows : [];
    const noneCount = rows.filter((r) => r.prediction === NONE_LABEL).length;
    const familyCount = rows.filter(
        (r) => r.prediction && r.prediction !== NONE_LABEL,
    ).length;
    const distinctFamilies = new Set(
        rows.filter((r) => r.prediction && r.prediction !== NONE_LABEL).map((r) => r.prediction),
    );
    const withHistory = rows.filter((r) => r.transaction_count > 0).length;

    return {
        clients: rows.length,
        familyCount,
        distinctFamilies: distinctFamilies.size,
        noneCount,
        withHistory,
        withoutHistory: rows.length - withHistory,
    };
}

/**
 * Per-currency summary for one client. Currencies are NEVER merged.
 */
export function summariseByCurrency(history) {
    const map = new Map();

    (Array.isArray(history) ? history : []).forEach((t) => {
        const entry = map.get(t.currency) ?? {
            currency: t.currency,
            count: 0,
            inflow: 0,
            outflow: 0,
            fees: 0,
            first: t.timestamp,
            last: t.timestamp,
        };

        entry.count += 1;
        entry.fees += Number.isFinite(t.fee) ? t.fee : 0;

        const dir = String(t.direction).toLowerCase();
        if (dir === 'inbound' || dir === 'in' || dir === 'credit') entry.inflow += t.amount;
        else entry.outflow += t.amount;

        if (t.timestamp < entry.first) entry.first = t.timestamp;
        if (t.timestamp > entry.last) entry.last = t.timestamp;

        map.set(t.currency, entry);
    });

    return [...map.values()].sort((a, b) => b.count - a.count);
}

/**
 * Descriptive recurrence context, grouped by the EXACT description string
 * and currency. No description is ever mapped to a merchant family: doing so
 * would require a documented mapping, which is not supplied with the data.
 */
export function buildDescriptionContext(history) {
    const map = new Map();

    (Array.isArray(history) ? history : []).forEach((t) => {
        const key = `${t.description}\u0000${t.currency}`;
        const entry = map.get(key) ?? {
            description: t.description,
            currency: t.currency,
            count: 0,
            timestamps: [],
        };
        entry.count += 1;
        entry.timestamps.push(t.timestamp);
        map.set(key, entry);
    });

    return [...map.values()]
        .map((entry) => {
            const stamps = entry.timestamps.slice().sort();
            const gaps = [];
            for (let i = 1; i < stamps.length; i += 1) {
                gaps.push(daysBetween(stamps[i - 1], stamps[i]));
            }

            const medianGap = gaps.length > 0
                ? (() => {
                    const sorted = gaps.slice().sort((a, b) => a - b);
                    const mid = Math.floor(sorted.length / 2);
                    return sorted.length % 2 === 0
                        ? Math.round((sorted[mid - 1] + sorted[mid]) / 2)
                        : sorted[mid];
                })()
                : null;

            return {
                description: entry.description,
                currency: entry.currency,
                count: entry.count,
                first_seen: stamps[0],
                last_seen: stamps[stamps.length - 1],
                median_gap_days: medianGap,
                min_gap_days: gaps.length > 0 ? Math.min(...gaps) : null,
                max_gap_days: gaps.length > 0 ? Math.max(...gaps) : null,
            };
        })
        .sort((a, b) => b.count - a.count || b.last_seen.localeCompare(a.last_seen));
}

/**
 * Monthly activity counts for the timeline. Counts only \u2014 no amounts, so
 * different currencies are never implicitly combined.
 */
export function buildMonthlyTimeline(history) {
    const map = new Map();

    (Array.isArray(history) ? history : []).forEach((t) => {
        const month = t.timestamp.slice(0, 7);
        map.set(month, (map.get(month) ?? 0) + 1);
    });

    const months = [...map.keys()].sort();
    if (months.length === 0) return [];

    // Fill the gaps so a pause in activity is visible rather than hidden.
    const out = [];
    const [startY, startM] = months[0].split('-').map(Number);
    const [endY, endM] = months[months.length - 1].split('-').map(Number);

    let y = startY;
    let m = startM;
    let guard = 0;

    while ((y < endY || (y === endY && m <= endM)) && guard < 600) {
        const key = `${y}-${String(m).padStart(2, '0')}`;
        out.push({ month: key, count: map.get(key) ?? 0 });
        m += 1;
        if (m > 12) {
            m = 1;
            y += 1;
        }
        guard += 1;
    }

    return out;
}

export function uniqueValues(history, field) {
    const set = new Set();
    (Array.isArray(history) ? history : []).forEach((t) => {
        if (t[field]) set.add(t[field]);
    });
    return [...set].sort();
}
