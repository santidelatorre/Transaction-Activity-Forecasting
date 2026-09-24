import { useEffect, useState } from 'react';
import { Panel } from '@/components/insights/Primitives';

function score(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '—';
}

export default function ExperimentHistory() {
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);
    const [offset, setOffset] = useState(0);
    const [refresh, setRefresh] = useState(0);
    const limit = 20;

    useEffect(() => {
        const controller = new AbortController();
        setData(null);
        setError(null);
        fetch(`/api/v1/experiments?limit=${limit}&offset=${offset}`, { signal: controller.signal })
            .then(async (response) => {
                if (!response.ok) throw new Error('Experiment history is currently unavailable.');
                return response.json();
            })
            .then(setData)
            .catch((reason) => {
                if (reason.name !== 'AbortError') setError(reason.message);
            });
        return () => controller.abort();
    }, [offset, refresh]);

    return (
        <Panel title="Experiment history" description="Latest recorded runs. Compare scores only when their data and validation protocol match; selection is local to each run.">
            <div className="mb-4 flex gap-4">
                <button onClick={() => setRefresh((value) => value + 1)}>Refresh</button>
                <button disabled={offset === 0 || !data} onClick={() => setOffset(offset - limit)}>Previous</button>
                <button disabled={!data || data.experiments.length < limit} onClick={() => setOffset(offset + limit)}>Next</button>
            </div>
            {error && <p role="alert">{error}</p>}
            {!data && !error && <p>Loading experiment history…</p>}
            {data && data.experiments.length === 0 && <p>No recorded experiments on this page.</p>}
            {data?.experiments.map((row) => (
                <details key={row.experiment_id} className="mb-3 rounded-lg border p-3">
                    <summary>{row.model_name} · Macro-F1 {score(row.metrics.macro_f1)} · Accuracy {score(row.metrics.accuracy)} · {row.result}</summary>
                    <p>{row.recorded_at_utc} · {row.model_version} · Scope: {row.metrics.scope ?? 'Not recorded'}</p>
                    <p>Run: {row.metrics.run_id ?? 'Not recorded'} · Experiment: {row.experiment_id}</p>
                    <p>Commit: {row.git_commit ?? 'Unknown'} · Working tree: {row.git_dirty === null ? 'Unknown' : row.git_dirty ? 'Modified' : 'Clean'}</p>
                    <p>Compute: {row.compute_seconds ?? '—'} s · Baseline delta: {score(row.delta_vs_baseline)}</p>
                    <p>{row.notes}</p>
                    <p>{row.risk_notes}</p>
                    <pre className="overflow-auto text-xs">{JSON.stringify({ features: row.features, hyperparameters: row.hyperparameters, metrics: row.metrics }, null, 2)}</pre>
                </details>
            ))}
        </Panel>
    );
}
