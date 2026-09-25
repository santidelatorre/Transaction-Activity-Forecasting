import { useEffect, useState } from 'react';

const LIMIT = 8;

function score(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '—';
}

function actualWeights(row) {
    if (row.metrics?.alpha == null) return '—';
    const alpha = Number(row.metrics?.alpha);
    if (!row.model_name?.startsWith('ensemble_') || !Number.isFinite(alpha) || alpha < 0 || alpha > 1) return '—';
    return `ML ${((1 - alpha) * 100).toFixed(0)}% / heuristic ${(alpha * 100).toFixed(0)}%`;
}

export default function ExperimentHistoryTable() {
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);
    const [offset, setOffset] = useState(0);
    const [refresh, setRefresh] = useState(0);

    useEffect(() => {
        const controller = new AbortController();
        setData(null);
        setError(null);
        fetch(`/api/v1/experiments?limit=${LIMIT}&offset=${offset}`, { signal: controller.signal })
            .then(async (response) => {
                if (!response.ok) throw new Error('Experiment history is currently unavailable.');
                return response.json();
            })
            .then(setData)
            .catch((reason) => { if (reason.name !== 'AbortError') setError(reason.message); });
        return () => controller.abort();
    }, [offset, refresh]);

    return <section className="border border-[#EBEBEB] bg-white">
        <div className="flex flex-wrap items-end justify-between gap-3 border-b border-[#EBEBEB] p-5 sm:p-6">
            <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[#E60000]">Reproducibility</p>
                <h2 className="mt-2 text-xl font-semibold tracking-tight">Experiment history</h2>
                <p className="mt-1 text-xs leading-relaxed text-[#666]">Recorded VALID runs from /api/v1/experiments. Compare only runs using the same data and protocol.</p>
            </div>
            <button type="button" onClick={() => setRefresh((value) => value + 1)} className="border border-[#D6D6D6] px-3 py-2 text-xs font-semibold hover:border-[#E60000]">Refresh</button>
        </div>
        {error && <p role="alert" className="p-5 text-sm text-[#9C0000]">{error}</p>}
        {!data && !error && <p className="p-5 text-sm text-[#666]">Loading recorded runs…</p>}
        {data && data.experiments.length === 0 && <p className="p-5 text-sm text-[#666]">No recorded experiments on this page. Run the baseline to generate tracked validation runs.</p>}
        {data && data.experiments.length > 0 && <div className="overflow-x-auto"><table className="w-full min-w-[780px] text-left text-xs">
            <thead className="bg-[#F5F5F5] uppercase tracking-wider text-[#666]"><tr>
                <th className="px-5 py-3 font-semibold">Model / version</th><th className="px-5 py-3 font-semibold">Validation Macro-F1</th>
                <th className="px-5 py-3 font-semibold">Δ vs baseline</th><th className="px-5 py-3 font-semibold">Recorded weights</th><th className="px-5 py-3 font-semibold">Run</th>
            </tr></thead>
            <tbody>{data.experiments.map((row) => <tr key={row.experiment_id} className="border-t border-[#EBEBEB] align-top">
                <td className="px-5 py-3"><span className="font-medium">{row.model_name}</span><span className="mt-1 block font-mono text-[11px] text-[#666]">{row.model_version ?? 'Unspecified'}</span></td>
                <td className="px-5 py-3 font-mono tabular-nums">{score(row.metrics?.macro_f1)}</td>
                <td className="px-5 py-3 font-mono tabular-nums">{score(row.delta_vs_baseline)}</td>
                <td className="px-5 py-3 font-mono text-[11px]">{actualWeights(row)}</td>
                <td className="px-5 py-3 text-[#666]"><span className="block">{row.result}</span><span className="block font-mono text-[11px]">{row.recorded_at_utc?.slice(0, 10) ?? '—'}</span></td>
            </tr>)}</tbody>
        </table></div>}
        <div className="flex items-center justify-between border-t border-[#EBEBEB] px-5 py-3 text-xs text-[#666]">
            <span>Page {Math.floor(offset / LIMIT) + 1} · Weights shown only if recorded.</span>
            <div className="flex gap-2"><button type="button" disabled={offset === 0 || !data} onClick={() => setOffset((value) => value - LIMIT)} className="border border-[#D6D6D6] px-3 py-1.5 disabled:opacity-40">Previous</button>
                <button type="button" disabled={!data || data.experiments.length < LIMIT} onClick={() => setOffset((value) => value + LIMIT)} className="border border-[#D6D6D6] px-3 py-1.5 disabled:opacity-40">Next</button></div>
        </div>
    </section>;
}
