import { useEffect, useState } from 'react';
import { FlaskConical, Target } from 'lucide-react';
import InsightsLayout from '@/layouts/InsightsLayout';
import { Link } from '@/components/Link';
import { Button } from '@/components/ui/button';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table';
import { EmptyState, Note, Panel } from '@/components/insights/Primitives';
import { getResults } from '@/lib/api';
import ExperimentHistory from '@/components/insights/ExperimentHistory';

function score(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : 'Not provided';
}

export default function TechnicalResults() {
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        getResults().then(setResult).catch((reason) => setError(reason.message));
    }, []);

    const summary = result?.summary;
    const perClass = summary?.per_class ?? {};

    return (
        <InsightsLayout
            title="Technical results"
            subtitle="Validation metrics are read from the UBS runner output; test labels are never available to this dashboard."
        >
            <ExperimentHistory />
            {error && <p role="alert" className="rounded-lg bg-rose-50 p-4 text-sm text-rose-800">{error}</p>}
            {!result && !error && <p className="py-10 text-center text-sm text-slate-500">Loading evaluation artifacts…</p>}
            {result && !result.available ? (
                <Panel>
                    <EmptyState
                        icon={FlaskConical}
                        title="No evaluation artifacts found"
                        message="Run the UBS baseline to create the validation summary and per-class metrics."
                        action={<Button asChild variant="outline"><Link href="/insights/data">View submission status</Link></Button>}
                    />
                </Panel>
            ) : summary ? (
                <div className="space-y-6">
                    <div className="flex flex-wrap items-center gap-3">
                        <span className="rounded-lg bg-slate-100 px-3 py-1.5 font-mono text-sm font-medium text-[#0B2545]">{summary.best_model}</span>
                        <span className="rounded-lg bg-slate-100 px-3 py-1.5 text-xs font-medium text-slate-600">Scope: validation</span>
                        <span className="rounded-lg bg-slate-100 px-3 py-1.5 text-xs font-medium text-slate-600">{summary.valid_clients} validation clients</span>
                    </div>

                    <div className="grid gap-4 sm:grid-cols-2">
                        <div className="rounded-2xl border border-teal-200 bg-teal-50/60 p-5 shadow-sm">
                            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Macro-F1</p>
                            <p className="mt-2.5 font-mono text-3xl font-semibold tabular-nums text-[#0B2545]">{score(summary.best_macro_f1)}</p>
                            <p className="mt-2 text-xs text-slate-500">Official eight-class validation metric.</p>
                        </div>
                        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Accuracy</p>
                            <p className="mt-2.5 font-mono text-3xl font-semibold tabular-nums text-[#0B2545]">{score(summary.best_accuracy)}</p>
                            <p className="mt-2 text-xs text-slate-500">Secondary validation metric.</p>
                        </div>
                    </div>

                    <Note>These scores describe the validation split used during model selection. They are not test-set or leaderboard scores.</Note>

                    <Panel title="Per-class scores" description="Values produced by the shared UBS evaluation code.">
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Class</TableHead>
                                    <TableHead className="text-right">Precision</TableHead>
                                    <TableHead className="text-right">Recall</TableHead>
                                    <TableHead className="text-right">F1</TableHead>
                                    <TableHead className="text-right">Support</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {(result.labels ?? Object.keys(perClass)).map((label) => {
                                    const values = perClass[label] ?? {};
                                    return (
                                        <TableRow key={label}>
                                            <TableCell className="font-mono text-sm font-medium">{label}</TableCell>
                                            <TableCell className="text-right font-mono">{score(values.precision)}</TableCell>
                                            <TableCell className="text-right font-mono">{score(values.recall)}</TableCell>
                                            <TableCell className="text-right font-mono">{score(values.f1 ?? values['f1-score'])}</TableCell>
                                            <TableCell className="text-right font-mono">{values.support ?? 'Not provided'}</TableCell>
                                        </TableRow>
                                    );
                                })}
                            </TableBody>
                        </Table>
                    </Panel>

                    <Panel title="Model comparisons" description="Candidates evaluated on validation before selecting the final model.">
                        {result.experiments.length === 0 ? (
                            <EmptyState icon={Target} title="No experiment table found" />
                        ) : (
                            <Table>
                                <TableHeader>
                                    <TableRow><TableHead>Model</TableHead><TableHead className="text-right">Macro-F1</TableHead><TableHead className="text-right">Accuracy</TableHead><TableHead className="text-right">Train time</TableHead></TableRow>
                                </TableHeader>
                                <TableBody>
                                    {result.experiments.map((row, index) => (
                                        <TableRow key={`${row.model ?? row.name ?? 'model'}-${index}`}>
                                            <TableCell className="font-medium">{row.model ?? row.name ?? row.model_name ?? 'Model'}</TableCell>
                                            <TableCell className="text-right font-mono">{score(Number(row.macro_f1))}</TableCell>
                                            <TableCell className="text-right font-mono">{score(Number(row.accuracy))}</TableCell>
                                            <TableCell className="text-right font-mono">{row.train_seconds ? `${Number(row.train_seconds).toFixed(2)} s` : '—'}</TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        )}
                    </Panel>
                </div>
            ) : null}
        </InsightsLayout>
    );
}
