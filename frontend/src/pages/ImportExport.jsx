import { useEffect, useState } from 'react';
import { Download, FileCheck2, FileOutput } from 'lucide-react';
import InsightsLayout from '@/layouts/InsightsLayout';
import { Button } from '@/components/ui/button';
import { EmptyState, Note, Panel } from '@/components/insights/Primitives';
import { getHealth } from '@/lib/api';

export default function ImportExport() {
    const [health, setHealth] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        getHealth().then(setHealth).catch((reason) => setError(reason.message));
    }, []);

    const ready = health?.predictions_available;

    return (
        <InsightsLayout
            title="Submission"
            subtitle="The dashboard reads the validated CSV created by the Python UBS baseline runner."
        >
            {error && <p role="alert" className="mb-5 rounded-lg bg-rose-50 p-4 text-sm text-rose-800">{error}</p>}
            <div className="space-y-6">
                <Panel title="Generated submission" description="Predictions are validated against the sample IDs and official label vocabulary before download.">
                    {ready ? (
                        <div className="flex flex-col gap-5 p-5 sm:flex-row sm:items-center sm:justify-between">
                            <div className="flex items-start gap-3">
                                <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-50 text-[#0F766E]">
                                    <FileCheck2 className="h-5 w-5" />
                                </span>
                                <div>
                                    <p className="text-sm font-semibold text-[#0B2545]">UBS V1 submission is ready</p>
                                    <p className="mt-1 text-xs text-slate-500">Source: outputs/predictions/submission_v1.csv</p>
                                </div>
                            </div>
                            <Button asChild className="bg-[#0F766E] text-white hover:bg-[#115e59]">
                                <a href="/api/v1/submission.csv" download>
                                    <Download className="h-4 w-4" /> Download CSV
                                </a>
                            </Button>
                        </div>
                    ) : (
                        <EmptyState
                            icon={FileOutput}
                            title="Submission has not been generated"
                            message="Place the UBS dataset under data/raw/ubs_2026, then run the baseline command below. The dashboard will load its validated predictions and metrics automatically."
                        />
                    )}
                </Panel>

                <Panel title="Generate or refresh the baseline" description="Run this once from the repository root; FastAPI serves the resulting artifacts without retraining on page requests.">
                    <div className="space-y-4 p-5">
                        <pre className="overflow-x-auto rounded-xl bg-[#0B2545] p-4 text-sm text-slate-50"><code>python scripts/run_ubs_baseline.py --config configs/ubs_v1.toml</code></pre>
                        <Note>The runner writes the submission to <code>outputs/predictions/submission_v1.csv</code> and validation reports to <code>outputs/metrics/ubs_v1/</code>.</Note>
                    </div>
                </Panel>
            </div>
        </InsightsLayout>
    );
}
