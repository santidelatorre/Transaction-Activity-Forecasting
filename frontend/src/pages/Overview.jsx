import { useEffect, useState } from 'react';
import { Activity, CircleSlash, Database, Layers, Search, Users } from 'lucide-react';
import InsightsLayout from '@/layouts/InsightsLayout';
import { Link } from '@/components/Link';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table';
import { EmptyState, LabelChip, Note, Panel, Stat } from '@/components/insights/Primitives';
import { getOverview } from '@/lib/api';
import { CUTOFF_DATE, HORIZON_DAYS } from '@/lib/insights/constants';

const PAGE_SIZE = 15;

function Distribution({ rows }) {
    const max = Math.max(...rows.map((row) => row.count), 1);
    return (
        <div className="space-y-3 p-5">
            {rows.map((row) => (
                <div key={row.label} className="flex items-center gap-3">
                    <span className="w-24 shrink-0 font-mono text-xs text-slate-600">{row.label}</span>
                    <div className="h-6 flex-1 overflow-hidden rounded-md bg-slate-100">
                        <div
                            className={`h-full rounded-md ${row.label === 'none' ? 'bg-slate-400' : 'bg-[#0F766E]'}`}
                            style={{ width: `${(row.count / max) * 100}%` }}
                        />
                    </div>
                    <span className="w-24 shrink-0 text-right font-mono text-xs tabular-nums text-slate-600">
                        {row.count} · {(row.share * 100).toFixed(1)}%
                    </span>
                </div>
            ))}
        </div>
    );
}

export default function Overview() {
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
                .then((result) => {
                    if (active) {
                        setData(result);
                        setError(null);
                    }
                })
                .catch((reason) => {
                    if (active) setError(reason.message);
                })
                .finally(() => {
                    if (active) setLoading(false);
                });
        }, 180);
        return () => {
            active = false;
            window.clearTimeout(timer);
        };
    }, [page, query]);

    const totalPages = Math.max(Math.ceil((data?.total_filtered ?? 0) / PAGE_SIZE), 1);
    const summary = data?.summary;

    return (
        <InsightsLayout
            title="Overview"
            subtitle={`Client predictions from the UBS baseline. Cutoff ${CUTOFF_DATE}, ${HORIZON_DAYS}-day forecast horizon.`}
        >
            {error && <p role="alert" className="mb-5 rounded-lg bg-rose-50 p-4 text-sm text-rose-800">{error}</p>}
            {loading && !data && <p className="py-10 text-center text-sm text-slate-500">Loading baseline outputs…</p>}
            {data && !data.available ? (
                <Panel>
                    <EmptyState
                        icon={Database}
                        title="No baseline output found"
                        message={data.message}
                        action={
                            <Button asChild className="bg-[#0F766E] text-white hover:bg-[#115e59]">
                                <Link href="/insights/data">View submission status</Link>
                            </Button>
                        }
                    />
                </Panel>
            ) : summary ? (
                <>
                    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                        <Stat label="Clients" value={summary.clients} icon={Users} accent="navy" />
                        <Stat label="Recurring family predicted" value={summary.family_count} icon={Layers} accent="teal" />
                        <Stat label="Predicted none" value={summary.none_count} icon={CircleSlash} accent="slate" />
                        <Stat
                            label="Clients with history"
                            value={summary.with_history === null ? 'Unavailable' : `${summary.with_history} / ${summary.clients}`}
                            icon={Activity}
                            accent="amber"
                            hint={summary.without_history === null ? 'Local test transaction history is not available.' : `${summary.without_history} client(s) have no observed transactions.`}
                        />
                    </div>

                    {data.metrics && (
                        <div className="mt-4">
                            <Note>
                                Selected model: <strong>{data.metrics.best_model}</strong> · validation macro-F1{' '}
                                <strong>{Number(data.metrics.best_macro_f1).toFixed(4)}</strong>.
                            </Note>
                        </div>
                    )}
                    {!data.transactions_available && (
                        <div className="mt-4">
                            <Note>Predictions are available, but test_transactions.jsonl is missing locally; history counts are shown as unavailable input.</Note>
                        </div>
                    )}

                    <div className="mt-6 grid gap-6 lg:grid-cols-3">
                        <div className="lg:col-span-2">
                            <Panel
                                title="Clients"
                                description="Predictions are read from the validated baseline submission."
                                actions={
                                    <div className="relative">
                                        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                                        <Input
                                            value={query}
                                            onChange={(event) => {
                                                setQuery(event.target.value);
                                                setPage(1);
                                            }}
                                            placeholder="Search clients or labels"
                                            className="h-9 w-full pl-9 text-sm sm:w-64"
                                        />
                                    </div>
                                }
                            >
                                {data.clients.length === 0 ? (
                                    <EmptyState title="No matching clients" message="Try another client ID or label." />
                                ) : (
                                    <Table>
                                        <TableHeader>
                                            <TableRow>
                                                <TableHead>Client ID</TableHead>
                                                <TableHead>Prediction</TableHead>
                                                <TableHead className="text-right">Transactions</TableHead>
                                                <TableHead>Last observed</TableHead>
                                                <TableHead className="text-right">History</TableHead>
                                            </TableRow>
                                        </TableHeader>
                                        <TableBody>
                                            {data.clients.map((row) => (
                                                <TableRow key={row.client_id}>
                                                    <TableCell className="font-mono text-sm font-medium">{row.client_id}</TableCell>
                                                    <TableCell><LabelChip label={row.prediction} /></TableCell>
                                                    <TableCell className="text-right font-mono text-sm tabular-nums text-slate-600">{row.transaction_count ?? '—'}</TableCell>
                                                    <TableCell className="font-mono text-xs tabular-nums text-slate-600">{row.last_observed?.slice(0, 10) ?? '—'}</TableCell>
                                                    <TableCell className="text-right">
                                                        <Link
                                                            href={`/insights/client?client_id=${encodeURIComponent(row.client_id)}`}
                                                            className="text-sm font-medium text-[#0F766E] hover:underline"
                                                        >
                                                            View history
                                                        </Link>
                                                    </TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                )}
                                <div className="flex items-center justify-between border-t border-slate-200 px-5 py-3 text-xs text-slate-500">
                                    <span>{data.total_filtered} clients · page {page} of {totalPages}</span>
                                    <div className="flex gap-2">
                                        <Button variant="outline" size="sm" disabled={page <= 1 || loading} onClick={() => setPage((value) => value - 1)}>Previous</Button>
                                        <Button variant="outline" size="sm" disabled={page >= totalPages || loading} onClick={() => setPage((value) => value + 1)}>Next</Button>
                                    </div>
                                </div>
                            </Panel>
                        </div>
                        <Panel title="Prediction distribution" description="Counts come from the generated test submission.">
                            <Distribution rows={data.distribution} />
                        </Panel>
                    </div>
                </>
            ) : null}
        </InsightsLayout>
    );
}
