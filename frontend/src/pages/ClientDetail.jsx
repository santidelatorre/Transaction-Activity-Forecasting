import { useEffect, useMemo, useState } from 'react';
import { ArrowLeft, CalendarRange, Clock, Info, Repeat, SearchX, Wallet } from 'lucide-react';
import InsightsLayout from '@/layouts/InsightsLayout';
import { Link } from '@/components/Link';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { EmptyState, LabelChip, Note, Panel } from '@/components/insights/Primitives';
import { getClient } from '@/lib/api';
import {
    buildDescriptionContext,
    buildMonthlyTimeline,
    formatAmount,
    formatDate,
    formatDateTime,
    summariseByCurrency,
    uniqueValues,
} from '@/lib/insights/analyse';
import {
    CUTOFF_DATE,
    HORIZON_DAYS,
    HORIZON_END_DATE,
    NONE_EXPLANATION,
    NONE_LABEL,
} from '@/lib/insights/constants';

const ALL = '__all__';

function Timeline({ points }) {
    if (points.length === 0) return null;
    const max = Math.max(...points.map((p) => p.count), 1);

    return (
        <div className="px-5 py-5">
            <div className="flex items-end gap-1 overflow-x-auto pb-2" style={{ minHeight: 120 }}>
                {points.map((p) => (
                    <div key={p.month} className="flex min-w-[26px] flex-1 flex-col items-center gap-1.5" title={`${p.month}: ${p.count} transaction(s)`}>
                        <span className="font-mono text-[10px] tabular-nums text-slate-400">{p.count || ''}</span>
                        <div
                            className={`w-full rounded-t ${p.count > 0 ? 'bg-[#0F766E]' : 'bg-slate-200'}`}
                            style={{ height: `${Math.max((p.count / max) * 80, p.count > 0 ? 6 : 2)}px` }}
                        />
                        <span className="origin-top-left -rotate-45 whitespace-nowrap font-mono text-[9px] text-slate-400">
                            {p.month.slice(2)}
                        </span>
                    </div>
                ))}
            </div>
            <p className="mt-6 text-xs text-slate-500">
                Observed transaction counts per month. Counts only — amounts are not summed here, so currencies are never
                combined.
            </p>
        </div>
    );
}

export default function ClientDetail() {
    const clientId = useMemo(() => {
        if (typeof window === 'undefined') return '';
        const query = window.location.hash.split('?')[1] ?? '';
        return new URLSearchParams(query).get('client_id') ?? '';
    }, []);

    const [currency, setCurrency] = useState(ALL);
    const [direction, setDirection] = useState(ALL);
    const [detail, setDetail] = useState(null);
    const [loadError, setLoadError] = useState(null);

    useEffect(() => {
        let active = true;
        if (!clientId) return undefined;
        getClient(clientId)
            .then((result) => {
                if (active) {
                    setDetail(result);
                    setLoadError(null);
                }
            })
            .catch((error) => {
                if (active) setLoadError(error.message);
            });
        return () => {
            active = false;
        };
    }, [clientId]);

    const known = detail?.client_id === clientId;
    const history = detail?.transactions ?? [];
    const prediction = detail?.prediction ?? null;

    const currencies = useMemo(() => uniqueValues(history, 'currency'), [history]);
    const directions = useMemo(() => uniqueValues(history, 'direction'), [history]);

    const filtered = useMemo(
        () =>
            history.filter(
                (t) =>
                    (currency === ALL || t.currency === currency) &&
                    (direction === ALL || t.direction === direction),
            ),
        [history, currency, direction],
    );

    const perCurrency = useMemo(() => summariseByCurrency(filtered), [filtered]);
    const descriptions = useMemo(() => buildDescriptionContext(filtered), [filtered]);
    const timeline = useMemo(() => buildMonthlyTimeline(filtered), [filtered]);

    if (clientId && !detail && !loadError) {
        return <InsightsLayout title="Client details"><p className="text-sm text-slate-500">Loading client history…</p></InsightsLayout>;
    }

    if (!clientId || !known) {
        return (
            <InsightsLayout title="Client not found">
                <Panel>
                    <EmptyState
                        icon={SearchX}
                        title="No such client in the current dataset"
                        message={
                            clientId
                                ? loadError ?? `Client "${clientId}" is not part of the generated submission.`
                                : 'No client was specified. Pick one from the overview table.'
                        }
                        action={
                            <Button asChild variant="outline">
                                <Link href="/insights/overview">
                                    <ArrowLeft className="mr-2 h-4 w-4" />
                                    Back to overview
                                </Link>
                            </Button>
                        }
                    />
                </Panel>
            </InsightsLayout>
        );
    }

    const isNone = prediction === NONE_LABEL;

    return (
        <InsightsLayout>
            <Link
                href="/insights/overview"
                className="mb-5 inline-flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-[#0F766E]"
            >
                <ArrowLeft className="h-4 w-4" />
                Back to overview
            </Link>

            <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
                <div>
                    <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Client</p>
                    <h1 className="mt-1 font-mono text-2xl font-semibold tracking-tight sm:text-3xl">{clientId}</h1>
                </div>
            </div>

            {!detail.transactions_available && (
                <Note>Test transaction history is not available locally. The prediction is still shown from the validated submission.</Note>
            )}

            <div className="grid gap-6 lg:grid-cols-3">
                <div className="space-y-6 lg:col-span-2">
                    <Panel title="Prediction">
                        <div className="space-y-4 p-5">
                            <div className="flex flex-wrap items-center gap-3">
                                <LabelChip label={prediction} className="px-3 py-1.5 text-sm" />
                                {isNone && (
                                    <span className="text-sm font-medium text-[#0B2545]">{NONE_EXPLANATION}</span>
                                )}
                            </div>

                            {isNone && (
                                <Note>
                                    <strong>none</strong> is a deliberate prediction, not missing data and not
                                    uncertainty. It states that no recurring merchant family is expected to appear in the
                                    forecast window below.
                                </Note>
                            )}

                            {!prediction && (
                                <Note>
                                    No prediction is available for this client in the loaded dataset. Nothing has been
                                    substituted in its place.
                                </Note>
                            )}

                            <div className="grid gap-3 sm:grid-cols-2">
                                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                                    <p className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                                        <CalendarRange className="h-3.5 w-3.5" /> Cutoff
                                    </p>
                                    <p className="mt-2 font-mono text-lg font-semibold tabular-nums">{CUTOFF_DATE}</p>
                                    <p className="mt-1 text-xs text-slate-500">History is observed up to this date.</p>
                                </div>
                                <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                                    <p className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                                        <Clock className="h-3.5 w-3.5" /> Horizon
                                    </p>
                                    <p className="mt-2 font-mono text-lg font-semibold tabular-nums">{HORIZON_DAYS} days</p>
                                    <p className="mt-1 text-xs text-slate-500">
                                        Window ends {HORIZON_END_DATE}.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </Panel>

                    <Panel
                        title="Activity timeline"
                        description="Observed history only. No future dates or amounts are shown, because none are supplied with a prediction."
                    >
                        {timeline.length === 0 ? (
                            <EmptyState
                                title="No activity to plot"
                                message="There are no transactions matching the current filters for this client."
                            />
                        ) : (
                            <Timeline points={timeline} />
                        )}
                    </Panel>

                    <Panel
                        title="Transactions"
                        description={`${filtered.length} of ${history.length} observed transaction(s) shown.`}
                        actions={
                            <div className="flex flex-wrap gap-2">
                                <Select value={currency} onValueChange={setCurrency}>
                                    <SelectTrigger className="h-9 w-[150px] text-sm">
                                        <SelectValue placeholder="Currency" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value={ALL}>All currencies</SelectItem>
                                        {currencies.map((c) => (
                                            <SelectItem key={c} value={c}>{c}</SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>

                                <Select value={direction} onValueChange={setDirection}>
                                    <SelectTrigger className="h-9 w-[150px] text-sm">
                                        <SelectValue placeholder="Direction" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value={ALL}>All directions</SelectItem>
                                        {directions.map((d) => (
                                            <SelectItem key={d} value={d}>{d}</SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>
                        }
                    >
                        {history.length === 0 ? (
                            <EmptyState
                                title={detail.transactions_available ? 'No transaction history' : 'Transaction history unavailable'}
                                message={detail.transactions_available
                                    ? 'This client has no observed transactions in the loaded dataset. The prediction above still stands on its own; nothing has been inferred here.'
                                    : 'The local test_transactions.jsonl file is missing, so no client history can be displayed.'}
                            />
                        ) : filtered.length === 0 ? (
                            <EmptyState
                                icon={SearchX}
                                title="No transactions match these filters"
                                message="Adjust the currency or direction filter to see this client's history."
                            />
                        ) : (
                            <div className="overflow-x-auto">
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead>Timestamp</TableHead>
                                            <TableHead>Description</TableHead>
                                            <TableHead>Type</TableHead>
                                            <TableHead>MCC</TableHead>
                                            <TableHead>Direction</TableHead>
                                            <TableHead className="text-right">Amount</TableHead>
                                            <TableHead className="text-right">Fee</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {filtered
                                            .slice()
                                            .reverse()
                                            .map((t, i) => (
                                                <TableRow key={`${t.timestamp}-${i}`}>
                                                    <TableCell className="whitespace-nowrap font-mono text-xs tabular-nums text-slate-600">
                                                        {formatDateTime(t.timestamp)}
                                                    </TableCell>
                                                    <TableCell className="max-w-[220px] truncate text-sm" title={t.description}>
                                                        {t.description || '—'}
                                                    </TableCell>
                                                    <TableCell className="text-xs text-slate-600">{t.type || '—'}</TableCell>
                                                    <TableCell className="font-mono text-xs text-slate-500">{t.mcc || '—'}</TableCell>
                                                    <TableCell className="text-xs text-slate-600">{t.direction}</TableCell>
                                                    <TableCell className="whitespace-nowrap text-right font-mono text-xs tabular-nums">
                                                        {formatAmount(t.amount, t.currency)}
                                                    </TableCell>
                                                    <TableCell className="whitespace-nowrap text-right font-mono text-xs tabular-nums text-slate-500">
                                                        {formatAmount(t.fee, t.currency)}
                                                    </TableCell>
                                                </TableRow>
                                            ))}
                                    </TableBody>
                                </Table>
                            </div>
                        )}
                    </Panel>
                </div>

                <div className="space-y-6">
                    <Panel
                        title="Per-currency summary"
                        description="Each currency is summarised separately. Values are never converted or added together."
                    >
                        {perCurrency.length === 0 ? (
                            <EmptyState icon={Wallet} title="Nothing to summarise" message="No transactions match the current filters." />
                        ) : (
                            <div className="divide-y divide-slate-100">
                                {perCurrency.map((c) => (
                                    <div key={c.currency} className="px-5 py-4">
                                        <div className="flex items-center justify-between">
                                            <span className="font-mono text-sm font-semibold">{c.currency}</span>
                                            <span className="font-mono text-xs tabular-nums text-slate-500">
                                                {c.count} tx
                                            </span>
                                        </div>
                                        <dl className="mt-2.5 space-y-1.5 text-xs">
                                            <div className="flex justify-between">
                                                <dt className="text-slate-500">Outbound total</dt>
                                                <dd className="font-mono tabular-nums">{c.outflow.toFixed(2)}</dd>
                                            </div>
                                            <div className="flex justify-between">
                                                <dt className="text-slate-500">Inbound total</dt>
                                                <dd className="font-mono tabular-nums">{c.inflow.toFixed(2)}</dd>
                                            </div>
                                            <div className="flex justify-between">
                                                <dt className="text-slate-500">Fees</dt>
                                                <dd className="font-mono tabular-nums">{c.fees.toFixed(2)}</dd>
                                            </div>
                                            <div className="flex justify-between">
                                                <dt className="text-slate-500">First / last</dt>
                                                <dd className="font-mono tabular-nums">
                                                    {formatDate(c.first)} → {formatDate(c.last)}
                                                </dd>
                                            </div>
                                        </dl>
                                    </div>
                                ))}
                            </div>
                        )}
                    </Panel>

                    <Panel
                        title="Recurrence context"
                        description="Descriptive statistics over observed descriptions. This is historical context, not a verified explanation of the model."
                    >
                        {descriptions.length === 0 ? (
                            <EmptyState icon={Repeat} title="No descriptions observed" message="No transactions match the current filters." />
                        ) : (
                            <div className="max-h-[420px] divide-y divide-slate-100 overflow-y-auto">
                                {descriptions.slice(0, 20).map((d) => (
                                    <div key={`${d.description}-${d.currency}`} className="px-5 py-3.5">
                                        <p className="truncate text-sm font-medium" title={d.description}>
                                            {d.description || '—'}
                                        </p>
                                        <p className="mt-1 font-mono text-[11px] tabular-nums text-slate-500">
                                            {d.count}× · {d.currency} · last {formatDate(d.last_seen)}
                                        </p>
                                        <p className="mt-1 text-[11px] text-slate-500">
                                            {d.median_gap_days === null
                                                ? 'Seen once — no gap to measure.'
                                                : `Typical gap ${d.median_gap_days} d (range ${d.min_gap_days}–${d.max_gap_days} d).`}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        )}
                        <div className="border-t border-slate-100 px-5 py-4">
                            <p className="flex gap-2 text-[11px] leading-relaxed text-slate-500">
                                <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                                Descriptions are shown exactly as recorded and are not mapped to any merchant family. A
                                documented mapping would be required to do that, and none is supplied with the data.
                            </p>
                        </div>
                    </Panel>
                </div>
            </div>
        </InsightsLayout>
    );
}
