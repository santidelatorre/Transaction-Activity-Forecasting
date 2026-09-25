// Challenge constants. These are fixed by the UBS Transaction Activity
// Forecasting challenge definition and must never be inferred from data.

export const CUTOFF_DATE = '2026-01-01';
export const HORIZON_DAYS = 90;

// Cutoff + 90 days, computed once so the UI never hard-codes a second date.
export const HORIZON_END_DATE = (() => {
    const d = new Date(`${CUTOFF_DATE}T00:00:00Z`);
    d.setUTCDate(d.getUTCDate() + HORIZON_DAYS);
    return d.toISOString().slice(0, 10);
})();

// The eight allowed labels. "none" is a real prediction: it means no recurring
// merchant family is predicted inside the horizon. It is NOT uncertainty,
// missing data, or an absence of future transactions.
export const RECURRING_FAMILIES = [
    'cloud',
    'gym',
    'insurance',
    'mobile',
    'music',
    'software',
    'streaming',
];

export const NONE_LABEL = 'none';

export const ALLOWED_LABELS = [...RECURRING_FAMILIES, NONE_LABEL];

export const NONE_EXPLANATION =
    'No recurring merchant family is predicted within this forecast horizon.';

// Required columns of the submission file, in the exact required order.
export const SUBMISSION_COLUMNS = ['client_id', 'predicted_next_recurring_merchant'];

// Required fields of every transaction record in the JSONL file.
export const TRANSACTION_FIELDS = [
    'client_id',
    'timestamp',
    'amount',
    'currency',
    'direction',
    'type',
    'mcc',
    'description',
    'fee',
];

export const FOOTER_TEXT =
    'Independent hackathon prototype for the UBS Transaction Activity Forecasting challenge.';

// Muted, accessible chips for each label. Navy/teal family only.
export const LABEL_STYLES = {
    cloud: 'bg-sky-50 text-sky-900 ring-sky-200',
    gym: 'bg-amber-50 text-amber-900 ring-amber-200',
    insurance: 'bg-indigo-50 text-indigo-900 ring-indigo-200',
    mobile: 'bg-teal-50 text-teal-900 ring-teal-200',
    music: 'bg-rose-50 text-rose-900 ring-rose-200',
    software: 'bg-violet-50 text-violet-900 ring-violet-200',
    streaming: 'bg-cyan-50 text-cyan-900 ring-cyan-200',
    none: 'bg-slate-100 text-slate-700 ring-slate-300',
};
