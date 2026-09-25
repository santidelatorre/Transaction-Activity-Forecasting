import { useEffect, useState } from 'react';
import { ArrowRight, Search } from 'lucide-react';
import InsightsLayout from '@/layouts/InsightsLayout';
import { Link } from '@/components/Link';
import { getOverview } from '@/lib/api';
import { CUTOFF_DATE, HORIZON_DAYS } from '@/lib/insights/constants';
import ExperimentHistory from '@/components/insights/ExperimentHistoryTable';

const PAGE_SIZE = 10;

function Metric({ label, value, detail, primary = false }) {
    return <div className={`min-h-40 border p-5 sm:p-6 ${primary ? 'border-[#E60000] bg-white' : 'border-[#EBEBEB] bg-[#F5F5F5]'}`}>
        <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[#666]">{label}</p>
        <p className={`mt-4 font-mono text-4xl font-semibold tabular-nums tracking-tight sm:text-5xl ${primary ? 'text-[#E60000]' : 'text-[#1A1A1A]'}`}>{value}</p>
        <p className="mt-3 text-xs leading-relaxed text-[#666]">{detail}</p>
    </div>;
}

function FeatureEvidence({ metrics }) {
    const name = String(metrics?.best_model ?? '').toLowerCase();
    const measured = name.includes('logistic') || name.includes('catboost');
    const features = measured ? (metrics?.top_features ?? [])
        .filter((row) => Number.isFinite(Number(row.importance)) && Number(row.importance) >= 0).slice(0, 6) : [];
    const max = Math.max(...features.map((row) => Number(row.importance)), 1e-9);
    return <section className="border border-[#EBEBEB] bg-white p-5 sm:p-6">
        <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[#E60000]">Model transparency</p>
        <h2 className="mt-2 text-xl font-semibold tracking-tight">What drives the model?</h2>
        <p className="mt-1 text-xs text-[#666]">Selected model: <span className="font-mono">{metrics?.best_model ?? 'No run'}</span></p>
        {features.length > 0 ? <>
            <div className="mt-6 space-y-4">{features.map((row) => <div key={row.feature}>
                <div className="mb-1.5 flex justify-between gap-4 text-xs">
                    <span className="min-w-0 truncate font-mono" title={row.feature}>{row.feature}</span>
                    <span className="shrink-0 font-mono tabular-nums text-[#666]">{Number(row.importance).toFixed(3)}</span>
                </div>
                <div className="h-1.5 bg-[#EBEBEB]" role="img" aria-label={`${row.feature}: importance ${Number(row.importance).toFixed(3)}`}>
                    <div className="h-full bg-[#E60000]" style={{ width: `${(Number(row.importance) / max) * 100}%` }} />
                </div>
            </div>)}</div>
            <p className="mt-6 text-xs leading-relaxed text-[#666]">Global importance from the fitted {name.startsWith('ensemble') ? 'ML component of the selected ensemble' : 'selected model'}. Its scale is model-specific; it is not a client-level explanation or calibrated probability.</p>
        </> : <p className="mt-6 border-l-2 border-[#E60000] pl-4 text-sm leading-relaxed text-[#555]">Measured feature importance is unavailable for this selected model. We do not substitute illustrative weights or claim that 7/30-day features are the leading drivers.</p>}
    </section>;
}

function Distribution({ rows }) {
    const max = Math.max(...rows.map((row) => row.count), 1);
    return <section className="border border-[#EBEBEB] bg-white p-5 sm:p-6">
        <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[#E60000]">Portfolio view</p>
        <h2 className="mt-2 text-xl font-semibold tracking-tight">Predicted families</h2>
        <p className="mt-1 text-xs text-[#666]">Validated submission distribution, not ground truth.</p>
        <div className="mt-6 space-y-3">{rows.map((row) => <div key={row.label} className="grid grid-cols-[5.5rem_1fr_3rem] items-center gap-3 text-xs">
            <span className="font-mono">{row.label}</span>
            <div className="h-1.5 bg-[#EBEBEB]"><div className={`h-full ${row.label === 'none' ? 'bg-[#888]' : 'bg-[#E60000]'}`} style={{ width: `${(row.count / max) * 100}%` }} /></div>
            <span className="text-right font-mono tabular-nums text-[#666]">{row.count}</span>
        </div>)}</div>
    </section>;
}

export default function ExecutiveDashboard() {
    const [query, setQuery] = useState('');
    const [page, setPage] = useState(1);
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        let active = true;
        const timer = window.setTimeout(() => {
            setLoading(true);
            getOverview({ page, pageSize: PAGE_SIZE, query })
                .then((result) => { if (active) { setData(result); setError(null); } })
                .catch((reason) => { if (active) setError(reason.message); })
                .finally(() => { if (active) setLoading(false); });
        }, 180);
        return () => { active = false; window.clearTimeout(timer); };
    }, [page, query]);

    const summary = data?.summary;
    const metrics = data?.metrics;
    const totalPages = Math.max(Math.ceil((data?.total_filtered ?? 0) / PAGE_SIZE), 1);
    const macroF1 = Number(metrics?.best_macro_f1);
    const hasMacroF1 = metrics?.best_macro_f1 != null && Number.isFinite(macroF1);
    const familyRate = summary?.clients ? (summary.family_count / summary.clients) * 100 : null;

    return <InsightsLayout title="Recurring commitments, made visible" subtitle={`Observed transactions to ${CUTOFF_DATE} · ${HORIZON_DAYS}-day prediction horizon · one family per client`}>
        <div className="mb-8 border-l-4 border-[#E60000] bg-[#F5F5F5] px-5 py-4">
            <p className="text-sm font-medium">Potential recurring commitment detected.</p>
            <p className="mt-1 text-xs leading-relaxed text-[#666]">Find the predicted family, inspect transaction history, then assess uncertainty. A forecast is a decision aid, not a guaranteed payment.</p>
        </div>
        {error && <p role="alert" className="mb-5 border border-[#E60000] p-4 text-sm text-[#9C0000]">{error}</p>}
        {loading && !data && <p className="py-12 text-sm text-[#666]">Loading baseline outputs…</p>}
        {data && !data.available && <div className="border border-[#EBEBEB] bg-[#F5F5F5] p-6">
            <h2 className="text-lg font-semibold">No prediction artifact available</h2>
            <p className="mt-2 text-sm text-[#666]">{data.message}</p>
            <Link href="/insights/data" className="mt-4 inline-flex items-center gap-2 text-sm font-semibold text-[#E60000]">Check submission status <ArrowRight size={15} /></Link>
        </div>}
        {summary && <>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <Metric primary label="Validation Macro-F1" value={hasMacroF1 ? macroF1.toFixed(4) : '—'} detail={hasMacroF1 ? 'Measured on VALID, not TEST. Team aspiration ≥0.80 is not the achieved result.' : 'Run the baseline for validation metrics.'} />
                <Metric label="Clients evaluated" value={metrics?.valid_clients ?? '—'} detail="Labeled VALID clients used for the reported Macro-F1." />
                <Metric label="Prediction coverage" value="100%" detail={`${summary.clients.toLocaleString()} of ${summary.clients.toLocaleString()} sample client IDs have a valid label; “none” counts.`} />
                <Metric label="Recurring-family rate" value={familyRate === null ? '—' : `${familyRate.toFixed(1)}%`} detail={`${summary.family_count.toLocaleString()} of ${summary.clients.toLocaleString()} clients predicted as a family other than “none”. Not accuracy.`} />
            </div>
            <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 border-y border-[#EBEBEB] py-3 text-xs text-[#666]">
                <span>Model: <strong className="font-mono text-[#1A1A1A]">{metrics?.best_model ?? 'UBS V1 output'}</strong></span>
                <span>Pipeline: <strong className="font-mono text-[#1A1A1A]">UBS V1</strong></span>
                <span>Scope: <strong className="text-[#1A1A1A]">VALID metric · TEST predictions</strong></span>
                {!data.transactions_available && <span className="text-[#9C0000]">Local test history unavailable</span>}
            </div>
            <div className="mt-8 grid gap-5 lg:grid-cols-2"><FeatureEvidence metrics={metrics} /><Distribution rows={data.distribution} /></div>
            <section className="mt-8 border border-[#EBEBEB] bg-white">
                <div className="flex flex-wrap items-end justify-between gap-4 border-b border-[#EBEBEB] p-5 sm:p-6">
                    <div><p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[#E60000]">From prediction to evidence</p>
                        <h2 className="mt-2 text-xl font-semibold tracking-tight">Explore a client</h2>
                        <p className="mt-1 text-xs text-[#666]">Open a real prediction and observed history. No TEST labels are displayed.</p></div>
                    <label className="relative block"><span className="sr-only">Search clients or predicted families</span>
                        <Search size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[#666]" />
                        <input value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} placeholder="Client ID or family" className="h-10 w-60 border border-[#D6D6D6] bg-white pl-9 pr-3 text-sm outline-none focus:border-[#E60000]" />
                    </label>
                </div>
                <div className="overflow-x-auto"><table className="w-full min-w-[580px] text-left text-sm">
                    <thead className="bg-[#F5F5F5] text-[11px] uppercase tracking-wider text-[#666]"><tr><th className="px-5 py-3 font-semibold">Client</th><th className="px-5 py-3 font-semibold">Prediction</th><th className="px-5 py-3 font-semibold">Observed transactions</th><th className="px-5 py-3 font-semibold">Next step</th></tr></thead>
                    <tbody>{data.clients.map((row) => <tr key={row.client_id} className="border-t border-[#EBEBEB]">
                        <td className="px-5 py-3 font-mono text-xs">{row.client_id}</td><td className="px-5 py-3 font-mono text-xs">{row.prediction}</td>
                        <td className="px-5 py-3 font-mono text-xs text-[#666]">{row.transaction_count ?? 'Unavailable'}</td>
                        <td className="px-5 py-3"><Link href={`/insights/client?client_id=${encodeURIComponent(row.client_id)}`} className="inline-flex items-center gap-1 text-xs font-semibold text-[#E60000] hover:underline">Inspect evidence <ArrowRight size={13} /></Link></td>
                    </tr>)}</tbody>
                </table>{data.clients.length === 0 && <p className="p-6 text-sm text-[#666]">No matching clients.</p>}</div>
                <div className="flex items-center justify-between border-t border-[#EBEBEB] px-5 py-3 text-xs text-[#666]">
                    <span>{data.total_filtered} clients · page {page} of {totalPages}</span>
                    <div className="flex gap-2"><button type="button" disabled={page <= 1 || loading} onClick={() => setPage((value) => value - 1)} className="border border-[#D6D6D6] px-3 py-1.5 disabled:opacity-40">Previous</button>
                        <button type="button" disabled={page >= totalPages || loading} onClick={() => setPage((value) => value + 1)} className="border border-[#D6D6D6] px-3 py-1.5 disabled:opacity-40">Next</button></div>
                </div>
            </section>
            <div className="mt-8"><ExperimentHistory /></div>
        </>}
    </InsightsLayout>;
}
