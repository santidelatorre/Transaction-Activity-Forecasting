import { useEffect, useState } from 'react';
import { ArrowRight, CheckCircle2, Fingerprint, Search, ShieldCheck, Sparkles } from 'lucide-react';
import { Panel, LoadingState, LabelChip } from '@/components/insights/Primitives';
import { request } from '@/lib/api';
import { buildMonthlyTimeline } from '@/lib/insights/analyse';

const number = (value, digits = 3) => value == null ? 'Unavailable' : Number(value).toFixed(digits);
const TOOL_LABELS = {
  predict_client: 'Read frozen prediction', inspect_history: 'Inspect observed history',
  inspect_data_quality: 'Check identity quality', inspect_candidate_streams: 'Inspect candidate streams',
  inspect_recurrence: 'Measure recurrence', compare_alternatives: 'Compare alternatives',
  show_model_metadata: 'Verify model provenance', show_global_metrics: 'Read measured metrics',
};
const FLAG_LABELS = {
  small_prediction_margin: 'Close alternatives', low_model_score: 'Low model score',
  identity_quality_degraded: 'Generic or unknown identity', multiple_candidate_families: 'Multiple recurring families',
  components_disagree: 'Model and heuristic disagree', missing_supporting_recurrence: 'Limited supporting recurrence',
};

function Timeline({ history }) {
  const points = buildMonthlyTimeline(history);
  const maximum = Math.max(...points.map(point => point.count), 1);
  return <div className="mt-5 border-t border-slate-100 pt-4">
    <p className="mb-3 text-xs font-medium text-slate-500">Observed activity · transactions per month</p>
    <div className="flex h-24 items-end gap-1" aria-label="Observed monthly transaction counts">
      {points.map(point => <div key={point.month} className="flex min-w-0 flex-1 flex-col items-center gap-2" title={`${point.month}: ${point.count} transactions`}>
        <div className="w-full rounded-t bg-teal-600/75" style={{ height: `${Math.max(2, point.count / maximum * 65)}px` }} />
        <span className="text-[9px] text-slate-500">{point.month.slice(5)}</span>
      </div>)}
    </div>
    <p className="mt-1 text-[10px] text-slate-400">{points[0]?.month} to {points.at(-1)?.month} · pre-cutoff observations only</p>
  </div>;
}

function Stream({ stream }) {
  return <div className="grid grid-cols-[1fr_auto] gap-4 border-b border-slate-100 py-3 last:border-0">
    <div className="min-w-0">
      <p className="truncate text-sm font-semibold" title={stream.description}>{stream.description}</p>
      <p className="mt-1 text-xs text-slate-500">{stream.count} observations · median gap {number(stream.median_gap_days, 1)} days · {stream.currency.toUpperCase()}</p>
      <p className="mt-1 text-[11px] text-slate-500">Historical mean {number(stream.amount_mean, 2)} {stream.currency.toUpperCase()} · amount CV {number(stream.amount_cv, 2)}</p>
    </div>
    <div className="text-right">
      <span className={`text-xs font-medium ${stream.strong_recurrence ? 'text-teal-700' : 'text-slate-500'}`}>{stream.strong_recurrence ? 'Regular history' : 'Limited regularity'}</span>
      <p className="mt-1 text-[11px] text-slate-500">Associated: {stream.associated_family}</p>
    </div>
  </div>;
}

function observationSummary(step) {
  const result = step.result;
  if (step.tool === 'predict_client') return `${result.predicted_family} · score ${number(result.score)} · margin ${number(result.margin)}`;
  if (step.tool === 'inspect_history') return `${result.length} observed transactions, strictly before cutoff.`;
  if (step.tool === 'inspect_data_quality') return `Generic descriptions ${number(result.generic_share == null ? null : result.generic_share * 100, 1)}%; unknown family association ${number(result.unknown_identity_share == null ? null : result.unknown_identity_share * 100, 1)}%.`;
  if (step.tool === 'inspect_candidate_streams') return `${result.length} repeated description/currency streams.`;
  if (step.tool === 'inspect_recurrence') return `${result.strong_streams} streams meet the documented recurrence rule.`;
  if (step.tool === 'compare_alternatives') return result.alternatives.map(item => `${item.family}: score ${number(item.score)}, ${item.associated_streams.length} associated streams`).join(' / ');
  if (step.tool === 'show_global_metrics') return `VALID Macro-F1 ${number(result.macro_f1, 6)}; reused validation.`;
  return `${result.model_version} · SHA ${result.base_sha}`;
}

export default function App() {
  const [boot, setBoot] = useState(null);
  const [clientId, setClientId] = useState(new URLSearchParams(window.location.search).get('client_id') || '');
  const [query, setQuery] = useState(clientId);
  const [client, setClient] = useState(null);
  const [agent, setAgent] = useState(null);
  const [investigating, setInvestigating] = useState(false);
  const [error, setError] = useState('');
  const [agentError, setAgentError] = useState('');
  const [run, setRun] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all(['/model', '/metrics', '/cases'].map(path => request(path, { signal: controller.signal })))
      .then(([model, metrics, cases]) => {
        setBoot({ model, metrics, cases });
        setClientId(current => current || cases.find(item => item.kind === 'clear')?.client_id || '');
      }).catch(reason => { if (reason.name !== 'AbortError') setError(reason.message); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!boot || !clientId) return;
    const controller = new AbortController();
    setClient(null); setAgent(null); setError(''); setAgentError(''); setRun(0); setQuery(clientId);
    const url = new URL(window.location.href);
    url.searchParams.set('client_id', clientId);
    window.history.replaceState(null, '', url);
    request(`/clients/${encodeURIComponent(clientId)}`, { signal: controller.signal })
      .then(setClient).catch(reason => { if (reason.name !== 'AbortError') setError(reason.message); });
    return () => controller.abort();
  }, [boot, clientId]);

  useEffect(() => {
    if (!run || !client) return;
    const controller = new AbortController();
    setInvestigating(true); setAgentError('');
    request(`/clients/${encodeURIComponent(client.prediction.client_id)}/investigate`, { method: 'POST', signal: controller.signal })
      .then(setAgent).catch(reason => { if (reason.name !== 'AbortError') setAgentError(reason.message); })
      .finally(() => { if (!controller.signal.aborted) setInvestigating(false); });
    return () => { controller.abort(); setInvestigating(false); };
  }, [run, client]);

  const p = client?.prediction;
  const quality = client?.data_quality;
  const supporting = client?.candidates.filter(stream => stream.associated_family === p.predicted_family) || [];
  const visibleStreams = [...supporting, ...(client?.candidates.filter(stream => stream.associated_family !== p.predicted_family) || [])];

  return <div className="min-h-screen bg-[#f5f7f8] text-[#173047]">
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-6 py-5">
        <div className="flex items-center gap-3"><span className="rounded-xl bg-teal-700 p-2 text-white"><Fingerprint size={22} /></span>
          <div><p className="text-lg font-semibold tracking-tight">Recurring Insights</p><p className="text-[11px] text-slate-500">From transaction history to a conversation worth having</p></div>
        </div>
        <span className="flex items-center gap-2 text-xs text-slate-500"><ShieldCheck size={15} /> Product V4 · model {boot?.model.model_version || 'unavailable'} · 90-day horizon</span>
      </div>
    </header>
    <main className="mx-auto max-w-6xl px-6 py-6">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap gap-2">{boot?.cases.map(item => <button key={item.kind} onClick={() => setClientId(item.client_id)} aria-pressed={clientId === item.client_id}
          className={`rounded-full border px-4 py-2 text-xs font-semibold ${clientId === item.client_id ? 'border-teal-700 bg-teal-700 text-white' : 'border-slate-200 bg-white hover:border-teal-600'}`}>
          {item.kind === 'clear' ? '01 · Clearer model signal' : '02 · Ambiguous signal'}</button>)}</div>
        <form className="flex gap-2" onSubmit={event => { event.preventDefault(); if (query.trim()) setClientId(query.trim()); }}>
          <label className="sr-only" htmlFor="client-search">Client ID</label>
          <input id="client-search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Client ID" className="w-36 rounded-lg border border-slate-200 bg-white px-3 py-2 font-mono text-xs" />
          <button aria-label="Open client" className="rounded-lg border border-slate-200 bg-white px-3"><Search size={16} /></button>
        </form>
      </div>
      {error && <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 p-5 text-sm text-rose-900">{error}<p className="mt-2">Check the API and prepared model bundle. No replacement data is displayed.</p></div>}
      {!error && !client && <LoadingState message="Loading the frozen predictor and observed history…" />}
      {client && <>
        <section className="mb-5 grid gap-6 rounded-2xl bg-[#123a46] p-6 text-white md:grid-cols-[1.4fr_1fr]">
          <div>
            <p className="text-xs font-medium uppercase tracking-[0.14em] text-teal-200">Client overview · {p.client_id}</p>
            <p className="mt-4 text-sm text-teal-100">{p.predicted_family === 'none' ? 'No recurring family predicted in the horizon' : 'Potential recurring commitment detected'}</p>
            <h1 className="mt-1 text-4xl font-semibold capitalize tracking-tight">{p.predicted_family}</h1>
            <p className="mt-3 max-w-lg text-xs leading-relaxed text-slate-200">Predicted next recurring family · 90 days after {p.cutoff}. {p.predicted_family === 'none' ? '“none” is a class, not an abstention or a guarantee.' : 'A family-level prediction; exact future dates and amounts are not estimated.'}</p>
          </div>
          <div className="rounded-xl bg-white/10 p-4">
            <div className="flex justify-between"><span className="text-xs text-teal-100">Model score</span><strong className="font-mono text-xl">{number(p.score)}</strong></div>
            <p className="mt-1 text-[11px] text-slate-200">Uncalibrated score · top-two margin {number(p.margin)}</p>
            <div className="mt-4 space-y-2">{p.top_alternatives.slice(0, 2).map(item => <div key={item.family} className="flex items-center justify-between text-xs"><span>{item.family}</span><span className="font-mono">{number(item.score)}</span></div>)}</div>
          </div>
        </section>

        <div className="grid gap-5 lg:grid-cols-[1.4fr_1fr]">
          <Panel title="Why this prediction deserves a look" description={`${p.evidence.transaction_count} observed transactions · ${p.evidence.supporting_stream_count} supporting streams meet the recurrence rule`}>
            <div className="px-5 pb-5">
              {visibleStreams.length ? visibleStreams.slice(0, 3).map(stream => <Stream key={`${stream.description}-${stream.currency}`} stream={stream} />) : <p className="py-5 text-sm text-slate-500">No repeated outbound card-payment streams found.</p>}
              <p className="mt-3 text-[11px] leading-relaxed text-slate-500">Descriptions are normalized observations. Family associations come from the frozen training map and do not verify merchant identity. Historical context is not a causal model explanation.</p>
              <Timeline history={client.history} />
              {visibleStreams.length > 3 && <details className="mt-3"><summary className="cursor-pointer text-xs text-teal-700">All {visibleStreams.length} candidate streams</summary>{visibleStreams.slice(3).map(stream => <Stream key={`${stream.description}-${stream.currency}`} stream={stream} />)}</details>}
            </div>
          </Panel>
          <div className="space-y-5">
            <Panel title="What remains uncertain" description="Identity and competing evidence affect how we interpret the signal.">
              <div className="space-y-3 p-5 text-xs">
                <div className="flex justify-between"><span>Generic payment descriptions</span><strong>{quality.generic_share == null ? 'Unavailable' : `${number(quality.generic_share * 100, 1)}%`}</strong></div>
                <div className="flex justify-between"><span>Unknown family association</span><strong>{quality.unknown_identity_share == null ? 'Unavailable' : `${number(quality.unknown_identity_share * 100, 1)}%`}</strong></div>
                <div className="flex justify-between"><span>Recurring candidate families</span><strong>{p.evidence.candidate_families.length}</strong></div>
                <p className="rounded-lg bg-amber-50 p-3 leading-relaxed text-amber-900">{p.component_disagreement ? `Components disagree: numeric model → ${p.component_choices.family_identity_model}; recurrence heuristic → ${p.component_choices.periodicity_heuristic}.` : 'The numeric model and recurrence heuristic agree. Agreement does not establish correctness.'}</p>
                <details><summary className="cursor-pointer text-slate-500">How evidence is measured</summary><p className="mt-2 leading-relaxed">{p.evidence.strong_recurrence_rule}. Regularity = 1 / (1 + gap standard deviation / median gap). Amounts are grouped by currency. Generic descriptions use an explicit exact-text list.</p></details>
              </div>
            </Panel>
            <div className="rounded-xl border border-slate-200 bg-white p-4 text-xs text-slate-500">
              <div className="flex items-center gap-2 font-medium text-slate-700"><CheckCircle2 size={15} /> Measured model performance</div>
              <p className="mt-2">VALID Macro-F1 <strong className="font-mono text-slate-800">{number(boot.metrics.macro_f1, 6)}</strong> · eight classes</p>
              <p className="mt-1 text-[11px]">VALID was reused during development. This is not an independent generalization estimate.</p>
            </div>
          </div>
        </div>

        <section className="mt-5 rounded-2xl border border-teal-200 bg-white p-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div><h2 className="flex items-center gap-2 text-sm font-semibold"><Sparkles size={16} className="text-teal-700" /> Agent investigation</h2><p className="mt-1 text-xs text-slate-500">Conditional policy · up to 8 read-only tools · no LLM configured</p></div>
            <button onClick={() => setRun(value => value + 1)} disabled={investigating} className="flex items-center gap-2 rounded-lg bg-teal-700 px-4 py-2.5 text-xs font-semibold text-white hover:bg-teal-800 disabled:opacity-50">{investigating ? 'Investigating…' : 'Investigate this prediction'}<ArrowRight size={14} /></button>
          </div>
          {agentError && <p role="alert" className="mt-3 text-sm text-rose-700">{agentError}</p>}
          {!agent && <p className="mt-4 text-xs leading-relaxed text-slate-500">The agent chooses what to inspect from the margin, identity quality, recurring candidates and component disagreement. It stops when available evidence is sufficient, exhausted, or the tool budget is reached.</p>}
          {agent && <div className="mt-5">
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">{agent.trace.map(step => <div key={step.step} className="rounded-lg bg-slate-50 p-3"><p className="text-xs font-semibold"><span className="mr-2 text-teal-700">{String(step.step).padStart(2, '0')}</span>{TOOL_LABELS[step.tool]}</p><p className="mt-1.5 text-[11px] leading-relaxed text-slate-500">{observationSummary(step)}</p></div>)}</div>
            <div className="mt-4 flex flex-wrap gap-2">{agent.uncertainty_flags.map(flag => <span key={flag} className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] text-amber-900">{FLAG_LABELS[flag]}</span>)}</div>
            <p className="mt-4 text-sm font-medium">{agent.conclusion}</p>
            <p className="mt-2 text-[11px] text-slate-500">Stopped: {agent.stop_reason.replaceAll('_', ' ')} · {agent.steps}/{agent.max_steps} tools · frozen prediction <LabelChip label={agent.predicted_family} /> · submission unchanged</p>
          </div>}
        </section>

        <details className="mt-5 rounded-xl border border-slate-200 bg-white p-4"><summary className="cursor-pointer text-xs text-slate-600">Inspect transaction facts ({client.history.length})</summary>
          <div className="mt-3 max-h-72 overflow-auto"><table className="w-full text-left text-xs"><thead><tr>{['Timestamp', 'Description', 'Amount', 'Currency', 'Direction', 'Type'].map(label => <th key={label} className="p-2 font-medium">{label}</th>)}</tr></thead><tbody>{client.history.map((row, i) => <tr key={i} className="border-t border-slate-100"><td className="whitespace-nowrap p-2 font-mono">{row.timestamp.slice(0, 10)}</td><td className="p-2">{row.description}</td><td className="p-2 font-mono">{number(row.amount, 2)}</td><td className="p-2">{row.currency}</td><td className="p-2">{row.direction}</td><td className="p-2">{row.type}</td></tr>)}</tbody></table></div>
        </details>
      </>}
      <footer className="mt-6 space-y-1 text-[10px] leading-relaxed text-slate-500">
        {boot && <><p>Frozen model {boot.model.model_version} · base SHA <code className="break-all">{boot.model.base_sha}</code> · fit {boot.model.fit_scope} · observations TEST</p><p>Cases selected by score margin and pre-cutoff evidence, without TEST labels. Dataset is the supplied challenge data. No balances, future exact amounts or guaranteed payment dates.</p></>}
      </footer>
    </main>
  </div>;
}
