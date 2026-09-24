import { AlertTriangle, Inbox, Loader2 } from 'lucide-react';
import { LABEL_STYLES } from '@/lib/insights/constants';

/** Prediction label chip. Renders the literal label, including "none". */
export function LabelChip({ label, className = '' }) {
    if (!label) {
        return (
            <span className={`inline-flex items-center rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-500 ring-1 ring-slate-200 ${className}`}>
                No prediction
            </span>
        );
    }

    const style = LABEL_STYLES[label] ?? 'bg-slate-100 text-slate-700 ring-slate-300';

    return (
        <span className={`inline-flex items-center rounded-full px-2.5 py-1 font-mono text-xs font-medium ring-1 ${style} ${className}`}>
            {label}
        </span>
    );
}

export function Panel({ title, description, actions, children, className = '' }) {
    return (
        <section className={`rounded-2xl border border-slate-200 bg-white shadow-sm ${className}`}>
            {(title || actions) && (
                <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-200 px-5 py-4">
                    <div>
                        {title && <h2 className="text-sm font-semibold tracking-tight text-[#0B2545]">{title}</h2>}
                        {description && <p className="mt-1 max-w-2xl text-xs text-slate-500">{description}</p>}
                    </div>
                    {actions}
                </div>
            )}
            {children}
        </section>
    );
}

export function EmptyState({ title, message, icon: Icon = Inbox, action }) {
    return (
        <div className="flex flex-col items-center justify-center px-6 py-14 text-center">
            <span className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-slate-100 text-slate-400">
                <Icon className="h-5 w-5" />
            </span>
            <p className="text-sm font-semibold text-[#0B2545]">{title}</p>
            {message && <p className="mt-2 max-w-md text-sm text-slate-500">{message}</p>}
            {action && <div className="mt-5">{action}</div>}
        </div>
    );
}

export function LoadingState({ message = 'Reading file\u2026' }) {
    return (
        <div className="flex items-center justify-center gap-3 px-6 py-12 text-sm text-slate-500">
            <Loader2 className="h-4 w-4 animate-spin text-[#0F766E]" />
            {message}
        </div>
    );
}

/** Actionable validation error list. Rows are never silently discarded. */
export function ErrorList({ title, errors = [], truncated = 0 }) {
    const items = Array.isArray(errors) ? errors : [];
    if (items.length === 0) return null;

    return (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4">
            <div className="flex items-start gap-2.5">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-600" />
                <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold text-rose-900">{title}</p>
                    <ul className="mt-2 space-y-1.5 text-xs text-rose-800">
                        {items.map((e, i) => (
                            <li key={i} className="font-mono leading-relaxed">
                                {e.line ? <span className="font-semibold">Line {e.line}: </span> : null}
                                {e.message}
                            </li>
                        ))}
                    </ul>
                    {truncated > 0 && (
                        <p className="mt-2 text-xs italic text-rose-700">
                            …and {truncated} further issue(s). Fix these first, then re-import.
                        </p>
                    )}
                    <p className="mt-3 text-xs text-rose-700">
                        Nothing was imported. The previous dataset is unchanged, and no rows were discarded or filled in
                        automatically.
                    </p>
                </div>
            </div>
        </div>
    );
}

export function Stat({ label, value, hint, icon: Icon, accent = 'teal' }) {
    const accents = {
        teal: 'bg-teal-50 text-[#0F766E]',
        navy: 'bg-slate-100 text-[#0B2545]',
        amber: 'bg-amber-50 text-amber-700',
        slate: 'bg-slate-100 text-slate-500',
    };

    return (
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-3">
                {Icon && (
                    <span className={`flex h-9 w-9 items-center justify-center rounded-lg ${accents[accent] ?? accents.teal}`}>
                        <Icon className="h-4 w-4" />
                    </span>
                )}
                <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
            </div>
            <p className="mt-3 font-mono text-2xl font-semibold tabular-nums tracking-tight text-[#0B2545]">{value}</p>
            {hint && <p className="mt-1.5 text-xs text-slate-500">{hint}</p>}
        </div>
    );
}

/** Small explanatory note used to keep claims honest across screens. */
export function Note({ children }) {
    return (
        <p className="rounded-lg border border-slate-200 bg-slate-50 px-3.5 py-2.5 text-xs leading-relaxed text-slate-600">
            {children}
        </p>
    );
}
