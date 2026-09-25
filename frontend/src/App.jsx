import { useEffect, useId, useState } from 'react';
import { ArrowDown, ArrowRight, ChevronDown, ChevronLeft, ChevronRight, RefreshCw } from 'lucide-react';
import ubsLogo from './ubs-logo.svg';

const RED = '#E60000';
const PAGE_SIZE = 6;
const format = (value, digits = 0) => typeof value === 'number' && Number.isFinite(value)
  ? value.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
  : '—';
const percent = (value) => Number.isFinite(value) ? `${format(value * 100, 1)}%` : '—';
const signed = (value) => Number.isFinite(value) ? `${value > 0 ? '+' : ''}${format(value, 4)}` : '—';
const families = { cloud: 'cloud services', gym: 'gyms', insurance: 'insurance', mobile: 'mobile services', music: 'music', software: 'software', streaming: 'streaming' };

function featureName(name) {
  let match = /^identity_(\w+)_(aliases|share|count)$/.exec(name);
  if (match) {
    const prefix = { aliases: 'Distinct descriptions linked to', share: 'Share of card payments linked to', count: 'Card payments linked to' }[match[2]];
    return `${prefix} ${families[match[1]] ?? match[1]}`;
  }
  match = /^mcc_(\d+)_(count|share)$/.exec(name);
  if (match) return `${match[2] === 'share' ? 'Share' : 'Number'} of transactions · MCC ${match[1]}`;
  match = /^calendar_([A-Z]{3})_client_count_daily_mean_(\d+)d$/.exec(name);
  if (match) return `Average daily transactions in ${match[1]} · past ${match[2]} days`;
  match = /^(transactions|outgoing|frequency)_last_(\d+)d$/.exec(name);
  if (match) return `${{ transactions: 'Transactions', outgoing: 'Outgoing transactions', frequency: 'Daily transaction frequency' }[match[1]]} · past ${match[2]} days`;
  return ({ n_transactions: 'Observed transactions', days_since_last_transaction: 'Days since last transaction', history_days: 'Length of transaction history', transactions_per_30d: 'Transactions per 30 days', repeated_description_count: 'Repeated descriptions', best_recurrence_score: 'Strongest recurrence signal' })[name] ?? name;
}

function useResource(path, revision) {
  const [state, setState] = useState({ data: null, loading: true, error: null });
  useEffect(() => {
    const controller = new AbortController();
    let timedOut = false;
    setState({ data: null, loading: true, error: null });
    const timer = window.setTimeout(() => { timedOut = true; controller.abort(); }, 20000);
    fetch(path, { signal: controller.signal, headers: { Accept: 'application/json' } })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Results could not be loaded (${response.status}).`);
        return response.json();
      })
      .then((data) => setState({ data, loading: false, error: null }))
      .catch((error) => {
        if (error.name !== 'AbortError' || timedOut) setState({ data: null, loading: false, error: timedOut ? 'The request timed out. Please retry.' : error.message });
      })
      .finally(() => window.clearTimeout(timer));
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [path, revision]);
  return state;
}

function Eyebrow({ children, className = '' }) {
  return <p className={`text-[10px] font-semibold uppercase tracking-[0.2em] ${className}`}>{children}</p>;
}

function Empty({ error, loading, children }) {
  return <div role={error ? 'alert' : 'status'} className="flex min-h-40 flex-col items-center justify-center px-6 py-10 text-center">
    <span className={`mb-4 h-8 w-8 rounded-full border border-[#DADADA] border-t-[#E60000] ${loading ? 'motion-safe:animate-spin' : ''}`} aria-hidden="true" />
    <p className="max-w-sm text-sm leading-6 text-[#686868]">{loading ? 'Loading recorded results…' : error || children}</p>
  </div>;
}

function MetricCard({ number, title, value, annotation, primary = false, children }) {
  return <article className={`flex min-h-[205px] flex-col rounded-sm border bg-white px-6 py-6 sm:px-7 ${primary ? 'border-[#E60000]' : 'border-[#E4E4E4]'}`}>
    <div className="flex min-h-10 items-start justify-between gap-3">
      <h3 className="max-w-[220px] text-sm font-medium leading-5 text-[#4D4D4D]">{title}</h3>
      <span className="font-mono text-[10px] text-[#999]">{number}</span>
    </div>
    <p className={`mt-5 text-[clamp(2.5rem,4.3vw,3.75rem)] font-medium leading-none tabular-nums tracking-[-0.055em] ${primary ? 'text-[#E60000]' : 'text-[#1A1A1A]'}`}>{value}</p>
    <p className="mt-4 text-xs leading-5 text-[#686868]">{annotation}</p>
    {children && <div className="mt-auto pt-4 text-[11px] text-[#686868]">{children}</div>}
  </article>;
}

function chartRows(data) {
  const history = (data?.version_history ?? []).map((row) => ({ ...row, selected: false }));
  const official = data?.mainline_v4?.official_valid ?? [];
  const source = data?.mainline_v4?.source?.protocol_path;
  const additions = official
    .filter((row) => row.id === 'stream-historical' || row.id === 'stream-clean')
    .map((row) => ({
      id: row.id,
      label: row.id === 'stream-clean' ? 'V4 frozen' : 'Stream hist.',
      macro_f1: row.macro_f1,
      scope: 'VALID',
      protocol: 'main-stream-valid',
      kind: 'measured',
      is_subversion: false,
      selected: row.status === 'selected_on_main',
      evidence: row.note,
      source,
    }));
  return [...history, ...additions];
}

function VersionChart({ data, loading, error }) {
  const chartId = useId();
  const rows = chartRows(data);
  const [activeId, setActiveId] = useState(null);
  const active = rows.find((row) => row.id === activeId) ?? rows.find((row) => row.selected && row.macro_f1 != null) ?? rows[0];
  const width = 850, height = 300, left = 44, right = 70, top = 24, bottom = 82;
  const axisMax = rows.some((row) => row.macro_f1 > 0.6) ? 0.8 : 0.6;
  const ticks = axisMax === 0.8 ? [0, 0.2, 0.4, 0.6, 0.8] : [0, 0.2, 0.4, 0.6];
  const x = (index) => left + (index / Math.max(rows.length - 1, 1)) * (width - left - right);
  const y = (score) => top + (axisMax - score) / axisMax * (height - top - bottom);
  const lineSegments = rows.slice(1).flatMap((row, index) => {
    const previous = rows[index];
    return row.macro_f1 != null && previous.macro_f1 != null && row.protocol === previous.protocol
      ? [{ from: previous, to: row, index }] : [];
  });
  const measured = rows.filter((row) => Number.isFinite(row.macro_f1));
  return <section id="progress" className="min-w-0 scroll-mt-24 rounded-sm border border-[#E4E4E4] bg-white">
    <div className="px-6 pt-6 sm:px-7">
      <Eyebrow className="text-[#858585]">01 / Project evidence</Eyebrow>
      <h2 className="mt-2 text-xl font-medium tracking-tight">How the measured Macro-F1 changed</h2>
      <p className="mt-2 text-xs leading-5 text-[#686868]">Official eight-class VALID results in report order. The last point is Stream Identity clean frozen on main: Macro-F1 0.6348.</p>
    </div>
    {loading || error || !rows.length || !measured.length ? <Empty loading={loading} error={error}>No verified version results are available.</Empty> : <>
      <div className="overflow-x-auto px-4 pt-5 sm:px-6">
        <svg viewBox={`0 0 ${width} ${height}`} className="min-w-[550px] w-full" role="img" aria-labelledby={`${chartId}-title ${chartId}-desc`}>
          <title id={`${chartId}-title`}>Macro-F1 across measured project versions</title>
          <desc id={`${chartId}-desc`}>V1, V2 and measured V3 variants use the official eight-class VALID protocol. The final points are the Stream Identity historical score and the clean frozen VALID score from main. Lines join only adjacent comparable measurements. This is not TEST performance.</desc>
          {ticks.map((tick) => <g key={tick}>
            <line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} stroke="#EAEAEA" strokeDasharray="2 4" />
            <text x={left - 10} y={y(tick) + 4} textAnchor="end" fill="#777" fontSize="11">{format(tick, 1)}</text>
          </g>)}
          {lineSegments.map(({ from, to, index }) => <line key={`${from.id}-${to.id}`} x1={x(index)} y1={y(from.macro_f1)} x2={x(index + 1)} y2={y(to.macro_f1)} stroke="#686868" strokeWidth="2" />)}
          {rows.map((row, index) => <g key={row.id}>
            {row.kind === 'decision' && <>
              <line x1={x(index)} x2={x(index)} y1={top + 12} y2={height - bottom} stroke="#C8C8C8" strokeDasharray="4 4" />
              <rect x={x(index) - 6} y={top + 38} width="12" height="12" fill="white" stroke="#777" strokeWidth="2" />
              <text x={x(index)} y={top + 26} textAnchor="middle" fill="#666" fontSize="10">No new score</text>
            </>}
            {Number.isFinite(row.macro_f1) && <>
              {row.selected && <circle cx={x(index)} cy={y(row.macro_f1)} r="11" fill={RED} fillOpacity="0.12" />}
              {row.is_subversion
                ? <rect x={x(index) - 5} y={y(row.macro_f1) - 5} width="10" height="10" transform={`rotate(45 ${x(index)} ${y(row.macro_f1)})`} fill={row.selected ? RED : 'white'} stroke={RED} strokeWidth="2" />
                : <circle cx={x(index)} cy={y(row.macro_f1)} r="5" fill={row.selected ? RED : 'white'} stroke={RED} strokeWidth="2" />}
              <text x={x(index)} y={y(row.macro_f1) - 15} textAnchor="middle" fill={row.selected ? RED : '#333'} fontSize="11" fontWeight={row.selected ? 'bold' : 'normal'}>{format(row.macro_f1, 4)}</text>
            </>}
            {row.kind === 'unavailable' && <text x={x(index)} y={top + 45} textAnchor="middle" fill="#999" fontSize="18">—</text>}
            <text x={x(index)} y={height - 48} textAnchor="middle" fill={row.selected ? RED : '#333'} fontSize="11" fontWeight={row.selected ? 'bold' : 'normal'}>{row.label}</text>
            <text x={x(index)} y={height - 31} textAnchor="middle" fill="#888" fontSize="10">{row.kind === 'decision' ? 'Retained' : row.scope}</text>
            <circle cx={x(index)} cy={row.kind === 'decision' ? top + 44 : Number.isFinite(row.macro_f1) ? y(row.macro_f1) : top + 44} r="17" fill="transparent" tabIndex="0" role="button" aria-label={`${row.label}: ${Number.isFinite(row.macro_f1) ? `Macro-F1 ${format(row.macro_f1, 4)} on ${row.scope}` : row.kind === 'decision' ? 'no new evaluation score' : 'result unavailable'}`} onMouseEnter={() => setActiveId(row.id)} onFocus={() => setActiveId(row.id)} onClick={() => setActiveId(row.id)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setActiveId(row.id); } }} className="cursor-pointer focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#1A1A1A]" />
          </g>)}
        </svg>
      </div>
      <div className="mx-6 flex flex-wrap items-center justify-between gap-3 border-y border-[#EAEAEA] py-3 text-xs sm:mx-7" aria-live="polite">
        <div><span className="font-medium">{active?.label}</span><span className="ml-2 text-[#777]">{active?.evidence || active?.note}</span><p className="mt-1 break-all text-[10px] text-[#888]">Source: {active?.source}</p></div>
        <span className="shrink-0 font-mono text-lg tabular-nums">{format(active?.macro_f1, 4)}</span>
      </div>
      <p className="px-6 py-4 text-[11px] leading-5 text-[#757575] sm:px-7">Source-report order, not a chronology. The V3 report’s V2 control, cohort size, and VALID file fingerprints are checked before joining its line to V2. VALID was reused in selection; no TEST score is shown.</p>
    </>}
  </section>;
}

function ImportanceChart({ data, loading, error }) {
  const rows = data?.importance?.slice(0, 6) ?? [];
  const max = Math.max(1e-9, ...rows.map((row) => row.importance));
  return <section id="explainability" className="min-w-0 scroll-mt-24 rounded-sm border border-[#E4E4E4] bg-white">
    <div className="px-6 pt-6 sm:px-7">
      <Eyebrow className="text-[#858585]">02 / Model signals</Eyebrow>
      <h2 className="mt-2 text-xl font-medium tracking-tight">What does the model use?</h2>
      <p className="mt-2 text-xs leading-5 text-[#686868]">Measured global importance in the CatBoost component, not a per-client explanation.</p>
    </div>
    {loading || error || !rows.length ? <Empty loading={loading} error={error}>Measured feature importance is unavailable in this environment.</Empty> : <>
      <div className="space-y-4 px-6 pb-4 pt-6 sm:px-7">
        {rows.map((row, index) => <div key={row.feature}>
          <div className="mb-2 flex items-start justify-between gap-3 text-xs leading-4">
            <span title={row.feature} className="max-w-[85%] text-[#4D4D4D]">{featureName(row.feature)}</span>
            <span className="shrink-0 tabular-nums text-[#686868]">{format(row.importance, 2)}</span>
          </div>
          <div className="h-1.5 bg-[#F0F0F0]" role="img" aria-label={`${featureName(row.feature)}: importance ${format(row.importance, 3)}`}>
            <div className={`h-full ${index === 0 ? 'bg-[#E60000]' : 'bg-[#737373]'}`} style={{ width: `${row.importance / max * 100}%` }} />
          </div>
        </div>)}
      </div>
      <details className="mx-6 border-t border-[#EAEAEA] py-3 sm:mx-7">
        <summary className="cursor-pointer text-[11px] font-medium text-[#555]">Original feature names and scope</summary>
        <p className="mt-3 text-[11px] leading-5 text-[#686868]">{data.importance_scope}. Description-to-family links are inferred, not known merchant identities.</p>
        <dl className="mt-3 space-y-2">{rows.map((row) => <div key={row.feature}><dt className="break-all font-mono text-[10px]">{row.feature}</dt><dd className="text-[11px] text-[#686868]">{featureName(row.feature)}</dd></div>)}</dl>
      </details>
      <p className="px-6 pb-5 text-[11px] leading-5 text-[#757575] sm:px-7">CatBoost importance scale. Selected V3-A uses {percent(data.ensemble_weights?.catboost)} CatBoost weight and {percent(data.ensemble_weights?.recurrence_heuristic)} periodicity-heuristic weight; these are not component scores.</p>
    </>}
  </section>;
}

function DotScore({ score, evidence, split, name, selected }) {
  if (!Number.isFinite(score)) return <span className="text-[11px] text-[#8A8A8A]">Not evaluated</span>;
  const reported = evidence?.includes('reported');
  return <div className="flex items-center gap-3">
    <div className="relative h-5 min-w-28 flex-1 border-b border-[#D7D7D7]" role="img" aria-label={`${name}, ${split} Macro-F1 ${format(score, 4)}, ${evidence}`}>
      {[0, 1, 2, 3].map((tick) => <span key={tick} className="absolute bottom-0 h-1.5 border-l border-[#D7D7D7]" style={{ left: `${tick * 100 / 3}%` }} />)}
      <span className={`absolute bottom-[-5px] h-2.5 w-2.5 -translate-x-1/2 rounded-full border-2 ${selected ? 'border-[#E60000] bg-[#E60000]' : reported ? 'border-[#686868] bg-white' : 'border-[#686868] bg-[#686868]'}`} style={{ left: `${Math.min(100, score / 0.6 * 100)}%` }} />
    </div>
    <span className="w-16 shrink-0 text-right font-mono text-xs tabular-nums">{format(score, 4)}</span>
    <span className="w-16 shrink-0 text-[10px] text-[#777]">{reported ? 'Reported' : 'Verified'}</span>
  </div>;
}

function ScoreBar({ score, max = 0.8 }) {
  if (!Number.isFinite(score)) return <span className="text-[11px] text-[#8A8A8A]">—</span>;
  return <div className="flex items-center gap-3">
    <div className="relative h-4 min-w-24 flex-1 border-b border-[#D7D7D7]">
      <span className="absolute bottom-[-4px] h-2 w-2 -translate-x-1/2 rounded-full bg-[#E60000]" style={{ left: `${Math.min(100, (score / max) * 100)}%` }} />
    </div>
    <span className="w-16 shrink-0 text-right font-mono text-xs tabular-nums">{format(score, 4)}</span>
  </div>;
}

function MainlineV4({ data, loading, error }) {
  const bundle = data?.mainline_v4;
  const official = bundle?.official_valid ?? [];
  const ablations = bundle?.ablations ?? [];
  const perClass = bundle?.per_class_valid ?? [];
  return <section id="mainline" className="mt-6 scroll-mt-24 rounded-sm border border-[#E60000] bg-white">
    <div className="px-6 pt-6 sm:px-7">
      <Eyebrow className="text-[#E60000]">Current main</Eyebrow>
      <h2 className="mt-2 text-xl font-medium tracking-tight">Stream Identity V4 on main</h2>
      <p className="mt-2 max-w-3xl text-xs leading-5 text-[#686868]">{bundle?.note || 'Loading the mainline freeze.'}</p>
      <p className="mt-1 break-all text-[10px] text-[#888]">Source commit {bundle?.source?.commit || '—'} · {bundle?.source?.protocol_path}</p>
    </div>
    {loading || error || !official.length ? <Empty loading={loading} error={error}>Mainline V4 results are not in this snapshot.</Empty> : <>
      <div className="overflow-x-auto px-6 py-6 sm:px-7">
        <table className="w-full min-w-[720px] text-left text-xs">
          <caption className="sr-only">Official VALID comparison published on main.</caption>
          <thead className="border-y border-[#E4E4E4] text-[10px] uppercase tracking-[0.09em] text-[#757575]"><tr><th className="py-3">Model</th><th className="py-3">VALID Macro-F1</th><th className="py-3 text-right">Accuracy</th><th className="py-3">Status</th></tr></thead>
          <tbody>{official.map((row) => <tr key={row.id} className="border-b border-[#ECECEC]">
            <th className="py-3 pr-4 font-medium" scope="row">{row.name}</th>
            <td className="py-3 pr-4"><ScoreBar score={row.macro_f1} /></td>
            <td className="py-3 text-right font-mono tabular-nums">{format(row.accuracy, 3)}</td>
            <td className="py-3 text-[11px] text-[#686868]">{row.status === 'selected_on_main' ? 'Frozen on main' : row.status === 'invalid_confirmation' ? 'Reproduced · confirmation invalid' : 'Historical reference'}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <div className="overflow-x-auto border-t border-[#EAEAEA] px-6 py-6 sm:px-7">
        <h3 className="text-sm font-medium">TRAIN-only ablations</h3>
        <p className="mt-1 text-[11px] leading-5 text-[#686868]">Macro-F1 on the clean TRAIN protocol. Medium and severe are stability checks, not the challenge score. Only legacy_half was opened on VALID.</p>
        <table className="mt-4 w-full min-w-[860px] text-left text-xs">
          <thead className="border-y border-[#E4E4E4] text-[10px] uppercase tracking-[0.09em] text-[#757575]"><tr><th className="py-3">Variant</th><th className="py-3">OOF</th><th className="py-3 text-right">Accuracy</th><th className="py-3 text-right">Δ vs full</th><th className="py-3 text-right">Medium</th><th className="py-3 text-right">Severe</th></tr></thead>
          <tbody>{ablations.map((row) => <tr key={row.id} className={`border-b border-[#ECECEC] ${row.status === 'selected_on_main' ? 'bg-[#FFF6F6]' : ''}`}>
            <th className="py-3 pr-3 font-medium" scope="row">{row.name}{row.status === 'selected_on_main' ? ' · frozen' : ''}</th>
            <td className="py-3 pr-3"><ScoreBar score={row.train_oof} /></td>
            <td className="py-3 text-right font-mono tabular-nums">{format(row.accuracy, 3)}</td>
            <td className="py-3 text-right font-mono tabular-nums">{signed(row.delta_vs_full)}</td>
            <td className="py-3 text-right font-mono tabular-nums">{format(row.medium_f1, 4)}</td>
            <td className="py-3 text-right font-mono tabular-nums">{format(row.severe_f1, 4)}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <div className="overflow-x-auto border-t border-[#EAEAEA] px-6 py-6 sm:px-7">
        <h3 className="text-sm font-medium">Frozen recipe · VALID by class</h3>
        <table className="mt-4 w-full min-w-[640px] text-left text-xs">
          <thead className="border-y border-[#E4E4E4] text-[10px] uppercase tracking-[0.09em] text-[#757575]"><tr><th className="py-3">Class</th><th className="py-3 text-right">Precision</th><th className="py-3 text-right">Recall</th><th className="py-3 text-right">F1</th><th className="py-3 text-right">Support</th><th className="py-3 text-right">Predicted</th></tr></thead>
          <tbody>{perClass.map((row) => <tr key={row.class} className="border-b border-[#ECECEC]">
            <th className="py-3 font-medium" scope="row">{row.class}</th>
            <td className="py-3 text-right font-mono tabular-nums">{format(row.precision, 3)}</td>
            <td className="py-3 text-right font-mono tabular-nums">{format(row.recall, 3)}</td>
            <td className="py-3 text-right font-mono tabular-nums">{format(row.f1, 4)}</td>
            <td className="py-3 text-right font-mono tabular-nums">{format(row.support)}</td>
            <td className="py-3 text-right font-mono tabular-nums">{format(row.predicted)}</td>
          </tr>)}</tbody>
        </table>
      </div>
    </>}
  </section>;
}

function V4Comparison({ data, loading, error }) {
  const rows = data?.v4_experiments ?? [];
  return <section id="comparison" className="mt-6 scroll-mt-24 rounded-sm border border-[#E4E4E4] bg-white">
    <div className="px-6 pt-6 sm:px-7">
      <Eyebrow className="text-[#858585]">03 / V4 research decision</Eyebrow>
      <h2 className="mt-2 text-xl font-medium tracking-tight">What the latest experiments actually showed</h2>
      <p className="mt-2 max-w-3xl text-xs leading-5 text-[#686868]">Clean official-target TRAIN OOF and reused VALID are separate evaluations. A higher TRAIN score did not automatically justify deployment.</p>
    </div>
    {loading || error || !rows.length ? <Empty loading={loading} error={error}>No recorded V4 experiment scores are available.</Empty> : <>
      <div className="overflow-x-auto px-6 py-6 sm:px-7">
        <div className="min-w-[790px]" role="group" aria-label="V4 experiment Macro-F1 comparison">
          <div className="grid grid-cols-[205px_1fr_1fr] gap-5 border-b border-[#E4E4E4] pb-3 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#686868]">
            <span>Experiment / decision</span><span>TRAIN OOF · clean</span><span>VALID · reused</span>
          </div>
          {rows.map((row, index) => <div key={row.id} className={`grid grid-cols-[205px_1fr_1fr] items-center gap-5 border-b border-[#ECECEC] py-4 ${index % 2 ? 'bg-[#FAFAFA]' : ''}`}>
            <div className="pl-1"><p className="text-xs font-medium">{row.name}</p><p className="mt-1 break-all text-[10px] text-[#888]">{row.branch}</p><p className={`mt-1 text-[10px] ${row.status === 'selected' ? 'font-semibold text-[#E60000]' : 'text-[#777]'}`}>{row.status === 'selected' ? 'Selected · retained' : 'Evaluated · not selected'}</p></div>
            <DotScore score={row.train_oof} evidence={row.train_evidence} split="TRAIN OOF" name={row.name} selected={row.status === 'selected'} />
            <DotScore score={row.valid} evidence={row.valid_evidence} split="reused VALID" name={row.name} selected={row.status === 'selected'} />
          </div>)}
          <div className="grid grid-cols-[205px_1fr_1fr] gap-5 pt-2 text-[10px] text-[#888]"><span /><span>Macro-F1 scale: 0 → 0.6</span><span>Macro-F1 scale: 0 → 0.6</span></div>
        </div>
      </div>
      <div className="grid gap-3 border-t border-[#EAEAEA] px-6 py-5 text-[11px] leading-5 text-[#686868] sm:grid-cols-2 sm:px-7">
        <p><span className="font-medium text-[#1A1A1A]">Why V3-A?</span> The V4 synthesis retained the frozen predictor after TRAIN-to-VALID transfer failures. VALID was reused, so it is secondary evidence, not an untouched final test.</p>
        <p><span className="font-medium text-[#1A1A1A]">Not on the clean-score axis:</span> {data.v4_diagnostics?.map((item) => `${item.name} ${format(item.macro_f1, 4)} (${item.scope})`).join('; ') || 'No separate stress result recorded'}. A stress score is not a competing clean model score.</p>
        <p className="sm:col-span-2">Open markers mean author-reported rather than independently reproduced. Merchant-intelligence and SVD OOF use transductive unsupervised components. The fixed A/ranker average has 50/50 weights, not component scores; it was not evaluated on VALID.</p>
        <p className="break-all text-[10px] text-[#888] sm:col-span-2">Source: {data.evidence_sources?.v4?.path} · synthesis commit {data.evidence_sources?.v4?.commit || 'unavailable'}</p>
      </div>
    </>}
  </section>;
}

function ExperimentRows({ row, ensembleWeights, index, expanded, toggle }) {
  const scope = row.metrics?.scope;
  const mix = row.candidate === 'A' && ensembleWeights
    ? `${percent(ensembleWeights.catboost)} CatBoost + ${percent(ensembleWeights.recurrence_heuristic)} heuristic weights`
    : row.candidate === 'ensemble' ? '50% V3 + 50% V2 weights' : '—';
  return <>
    <tr className={`border-t border-[#EEEEEE] ${index % 2 ? 'bg-[#FAFAFA]' : 'bg-white'}`}>
      <th scope="row" className="px-6 py-4 font-normal sm:pl-7"><button type="button" onClick={toggle} aria-expanded={expanded} className="flex items-center gap-2 text-left hover:text-[#E60000]"><ChevronDown size={13} className={`shrink-0 transition-transform ${expanded ? 'rotate-180' : ''}`} /><span><span className="block font-medium">{row.model_name}</span><span className="mt-1 block font-mono text-[10px] text-[#888]">{row.sequence ? `E${row.sequence}` : row.experiment_id?.slice(0, 8)} · {row.model_version}</span></span></button></th>
      <td className="px-4 py-4 text-[#686868]">{scope === 'validation' ? 'VALID' : scope || 'Unspecified'}</td>
      <td className="px-4 py-4 text-right font-medium tabular-nums">{format(row.metrics?.macro_f1, 4)}</td>
      <td className="px-4 py-4 text-right tabular-nums text-[#686868]">{percent(row.metrics?.accuracy)}</td>
      <td className="px-4 py-4 text-right tabular-nums text-[#686868]">{signed(row.delta_vs_baseline)}</td>
      <td className="px-4 py-4 text-[11px] text-[#686868]">{mix}</td>
      <td className="px-6 py-4 text-right sm:pr-7"><span className={`inline-block whitespace-nowrap px-2 py-1 text-[10px] ${row.result === 'selected' ? 'border border-[#FFD1D1] bg-[#FFF6F6] text-[#C90000]' : 'text-[#757575]'}`}>{row.result === 'selected' ? 'Selected' : 'Evaluated'}</span></td>
    </tr>
    {expanded && <tr className="border-t border-[#EEEEEE] bg-[#F7F7F7]"><td colSpan={7} className="px-7 py-5 text-xs leading-6 text-[#686868]">
      <p>{row.notes || 'No additional notes in the experiment record.'}</p>
      <p>Evaluated clients: {format(row.metrics?.validation_clients)} · Source: {row.source || 'Experiment ledger'}.</p>
      <p className="break-all font-mono text-[10px]">Evaluation commit: {row.git_commit || 'Not recorded'} · Run: {row.metrics?.run_id || 'Not recorded'}</p>
    </td></tr>}
  </>;
}

function ExperimentTable({ revision, ensembleWeights }) {
  const [page, setPage] = useState(0);
  const [expanded, setExpanded] = useState(null);
  const { data, loading, error } = useResource(`/api/v1/experiments?limit=${PAGE_SIZE + 1}&offset=${page * PAGE_SIZE}`, revision);
  const rows = data?.experiments?.slice(0, PAGE_SIZE) ?? [];
  const hasNext = (data?.experiments?.length ?? 0) > PAGE_SIZE;
  return <section id="experiments" className="mt-6 scroll-mt-24 rounded-sm border border-[#E4E4E4] bg-white">
    <div className="flex flex-wrap items-end justify-between gap-4 px-6 py-6 sm:px-7">
      <div><Eyebrow className="text-[#858585]">04 / Audit trail</Eyebrow><h2 className="mt-2 text-xl font-medium tracking-tight">The underlying V3 evaluation record</h2></div>
      <span className="border border-[#E4E4E4] px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-[#686868]">Report order · reused VALID</span>
    </div>
    {loading || error || !rows.length ? <Empty loading={loading} error={error}>No experiments are available on this page.</Empty> : <div className="overflow-x-auto">
      <table className="w-full min-w-[760px] text-left text-xs">
        <caption className="sr-only">Recorded V3 experiments with Macro-F1, accuracy, evaluation scope, and provenance.</caption>
        <thead className="border-y border-[#E4E4E4] bg-[#F5F5F5] text-[10px] font-medium uppercase tracking-[0.09em] text-[#757575]"><tr><th scope="col" className="px-6 py-3.5 sm:pl-7">Experiment / model</th><th scope="col" className="px-4 py-3.5">Evaluation</th><th scope="col" className="px-4 py-3.5 text-right">Macro-F1</th><th scope="col" className="px-4 py-3.5 text-right">Accuracy</th><th scope="col" className="px-4 py-3.5 text-right">Δ vs V2</th><th scope="col" className="px-4 py-3.5">Mix weights</th><th scope="col" className="px-6 py-3.5 text-right sm:pr-7">Status</th></tr></thead>
        <tbody>{rows.map((row, index) => <ExperimentRows key={row.experiment_id} row={row} ensembleWeights={ensembleWeights} index={index} expanded={expanded === row.experiment_id} toggle={() => setExpanded(expanded === row.experiment_id ? null : row.experiment_id)} />)}</tbody>
      </table>
    </div>}
    <div className="flex items-center justify-between gap-4 border-t border-[#E4E4E4] px-6 py-4 text-[11px] text-[#686868] sm:px-7">
      <span>Page {page + 1}<span className="hidden sm:inline"> · delta uses each record’s V2 baseline</span></span>
      <div className="flex gap-2">
        <button type="button" aria-label="Previous experiment page" disabled={page === 0 || loading} onClick={() => { setPage(page - 1); setExpanded(null); }} className="flex items-center gap-1 border border-[#DADADA] px-3 py-2 hover:bg-[#F5F5F5] disabled:cursor-not-allowed disabled:opacity-35"><ChevronLeft size={13} /> Previous</button>
        <button type="button" aria-label="Next experiment page" disabled={!hasNext || loading} onClick={() => { setPage(page + 1); setExpanded(null); }} className="flex items-center gap-1 border border-[#DADADA] px-3 py-2 hover:bg-[#F5F5F5] disabled:cursor-not-allowed disabled:opacity-35">Next <ChevronRight size={13} /></button>
      </div>
    </div>
  </section>;
}

export default function App() {
  const [revision, setRevision] = useState(0);
  const { data, loading, error } = useResource('/api/v1/dashboard', revision);
  const metrics = data?.metrics;
  const frozen = data?.mainline_v4?.official_valid?.find((row) => row.status === 'selected_on_main');
  const v2Main = data?.mainline_v4?.official_valid?.find((row) => row.id === 'main-v2');
  const recalls = data?.mainline_v4?.per_class_valid?.map((row) => row.recall).filter((value) => Number.isFinite(value)) ?? [];
  const macroF1 = frozen?.macro_f1 ?? metrics?.macro_f1;
  const accuracy = frozen?.accuracy ?? metrics?.accuracy;
  const macroRecall = recalls.length ? recalls.reduce((sum, value) => sum + value, 0) / recalls.length : metrics?.macro_recall;
  const clients = frozen ? 1000 : metrics?.validation_clients;
  const delta = frozen && v2Main ? frozen.macro_f1 - v2Main.macro_f1 : metrics?.delta_vs_baseline;
  const ready = Boolean(data?.available) || Number.isFinite(frozen?.macro_f1);
  useEffect(() => { document.title = 'Transaction Activity Forecasting · Recurring Insights'; document.documentElement.lang = 'en'; }, []);
  return <div className="min-h-screen bg-white font-sans text-[#1A1A1A] selection:bg-[#FFE2E2]">
    <a href="#content" className="sr-only z-50 bg-white p-4 focus:not-sr-only focus:fixed">Skip to content</a>
    <header className="sticky top-0 z-30 border-b border-[#EAEAEA] bg-white/95 backdrop-blur-sm">
      <div className="mx-auto flex h-[76px] max-w-[1320px] items-center justify-between gap-5 px-5 sm:px-8 lg:px-12">
        <a href="#content" className="flex shrink-0 items-center gap-3" aria-label="Recurring Insights home"><span className="grid h-8 w-8 grid-cols-2 items-end gap-[3px] border-b-[3px] border-[#E60000] pb-[3px]" aria-hidden="true"><span className="h-3 bg-[#E60000]" /><span className="h-6 bg-[#E60000]" /></span><span className="text-base font-semibold tracking-[-0.035em]">Recurring Insights<span className="mt-0.5 block text-[9px] font-medium uppercase tracking-[0.22em] text-[#888]">Transaction intelligence</span></span></a>
        <nav aria-label="Main navigation" className="hidden items-center gap-7 text-xs text-[#686868] md:flex"><a href="#quality" className="hover:text-[#E60000]">Overview</a><a href="#progress" className="hover:text-[#E60000]">Versions</a><a href="#mainline" className="hover:text-[#E60000]">Main V4</a><a href="#comparison" className="hover:text-[#E60000]">V4 research</a><a href="#experiments" className="hover:text-[#E60000]">Audit trail</a></nav>
        <div className="flex shrink-0 items-center gap-2 sm:gap-5"><img src={ubsLogo} alt="UBS logo" className="h-7 w-auto sm:h-8" /><button type="button" onClick={() => setRevision((value) => value + 1)} disabled={loading} className="flex items-center gap-2 border border-[#DADADA] px-3 py-2 text-[11px] hover:border-[#1A1A1A] disabled:opacity-50" aria-label="Refresh results"><RefreshCw size={12} className={loading ? 'motion-safe:animate-spin' : ''} /><span className="hidden sm:inline">Refresh</span></button></div>
      </div>
    </header>
    <main id="content" className="mx-auto max-w-[1320px] scroll-mt-24 px-5 pb-14 sm:px-8 lg:px-12">
      <section className="grid gap-10 pb-10 pt-10 sm:pt-14 lg:grid-cols-[1fr_260px] lg:items-center lg:gap-20">
        <div><Eyebrow className="text-[#E60000]">UBS challenge / Swiss AI Weeks</Eyebrow><h1 className="mt-5 max-w-[780px] text-[clamp(2.25rem,4.2vw,3.5rem)] font-medium leading-[1.1] tracking-[-0.045em]">Transaction Activity<br /><span className="text-[#777]">Forecasting</span></h1><p className="mt-5 max-w-[600px] text-sm leading-7 text-[#686868]">Transaction history becomes an evidence-backed signal about a client’s next recurring-payment family. The measurement, uncertainty, and model decision stay visible.</p><a href="#progress" className="mt-6 inline-flex items-center gap-3 text-xs font-semibold text-[#E60000]">Explore the evidence <ArrowDown size={14} /></a></div>
        <aside className="relative overflow-hidden bg-[#1A1A1A] px-7 py-6 text-white lg:py-8" aria-label="Prediction horizon"><Eyebrow className="text-[#B8B8B8]">Prediction horizon</Eyebrow><p className="mt-5 text-6xl font-light leading-none tracking-[-0.055em]">90<span className="ml-2 text-base font-normal tracking-normal text-[#B8B8B8]">days</span></p><div className="my-5 flex items-center gap-2" aria-hidden="true"><span className="h-1.5 w-1.5 rounded-full bg-[#E60000]" /><span className="h-px flex-1 bg-[#666]" /><ArrowRight size={13} className="text-[#E60000]" /></div><p className="text-[11px] leading-5 text-[#B8B8B8]">One recurring family per client.<br />History cutoff: 1 Jan 2026.</p></aside>
      </section>
      <section id="quality" className="scroll-mt-24 border-t border-[#EAEAEA] pt-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3"><Eyebrow className="text-[#757575]">Measured result · selected predictor</Eyebrow><span className="flex items-center gap-2 text-[11px] text-[#686868]"><span className={`h-1.5 w-1.5 rounded-full ${ready ? 'bg-[#1A1A1A]' : 'bg-[#B8B8B8]'}`} />{loading ? 'Loading evaluation' : frozen ? 'Stream Identity clean frozen · VALID after freeze' : ready ? `${data.model_version} retained in V4 · reused VALID` : 'Evaluation artifact unavailable'}</span></div>
        {error && <div role="alert" className="mb-4 border-l-2 border-[#E60000] bg-[#FFF6F6] px-4 py-3 text-sm text-[#8A2020]">{error} Use Refresh to retry.</div>}
        {!loading && !error && !ready && <p role="status" className="mb-4 border-l-2 border-[#DADADA] bg-[#F5F5F5] px-4 py-3 text-sm text-[#686868]">The local V3-A evaluation artifact is unavailable. Recorded V1/V2 report scores remain visible; missing V3 results are not filled in.</p>}
        <div className="grid gap-4 md:grid-cols-3">
          <MetricCard primary number="01" title="Balanced prediction quality" value={format(macroF1, 4)} annotation="Macro-F1 · each of eight classes has equal weight.">{delta != null ? `${signed(delta)} F1 points versus V2 on VALID` : 'No local selected-model score available'}</MetricCard>
          <MetricCard number="02" title="Average class detection" value={percent(macroRecall)} annotation="Macro recall · average recall across the eight classes.">Overall accuracy: {percent(accuracy)} · VALID.</MetricCard>
          <MetricCard number="03" title="Clients evaluated" value={format(clients)} annotation={frozen ? 'VALID clients in the Stream Identity clean evaluation.' : 'VALID clients counted from the official confusion matrix.'}>Seven recurring families plus “none”.</MetricCard>
        </div>
      </section>
      <div className="mt-6 space-y-6"><VersionChart data={data} loading={loading} error={error} /><ImportanceChart data={data} loading={loading} error={error} /></div>
      <MainlineV4 data={data} loading={loading} error={error} />
      <V4Comparison data={data} loading={loading} error={error} />
      <ExperimentTable revision={revision} ensembleWeights={data?.ensemble_weights} />
      <section id="method" className="mt-10 grid gap-7 border-y border-[#E4E4E4] py-8 md:grid-cols-[1.1fr_1fr_1fr]"><div><Eyebrow className="text-[#E60000]">Trust needs context</Eyebrow><h2 className="mt-3 max-w-xs text-xl font-medium leading-7 tracking-tight">A useful signal.<br />With its limits in view.</h2></div><div><h3 className="text-xs font-semibold">History precedes prediction</h3><p className="mt-2 text-xs leading-6 text-[#686868]">The predictor uses transactions before the cutoff. Descriptions can be ambiguous; family associations are learned with client separation.</p></div><div><h3 className="text-xs font-semibold">Validation is not a final test</h3><p className="mt-2 text-xs leading-6 text-[#686868]">VALID was reused during research. “None” means no recurring family is predicted within the horizon, not a guarantee of zero transactions.</p></div></section>
      <details className="group border-b border-[#E4E4E4] py-4 text-xs text-[#686868]"><summary className="flex cursor-pointer list-none items-center justify-between gap-3"><span>Model record and provenance</span><ChevronDown size={14} className="group-open:rotate-180" /></summary><dl className="mt-4 grid gap-4 text-[11px] sm:grid-cols-2"><div><dt className="font-medium text-[#1A1A1A]">Selected model</dt><dd className="mt-1">{data?.model_version || 'Unavailable'} · 90-day horizon</dd></div><div><dt className="font-medium text-[#1A1A1A]">Frozen baseline commit</dt><dd className="mt-1 break-all font-mono text-[10px]">{data?.base_sha || 'Unavailable'}</dd></div><div><dt className="font-medium text-[#1A1A1A]">Evaluation commit</dt><dd className="mt-1 break-all font-mono text-[10px]">{data?.evaluation_sha || 'Not recorded'}</dd></div><div><dt className="font-medium text-[#1A1A1A]">Evaluation report SHA-256</dt><dd className="mt-1 break-all font-mono text-[10px]">{data?.report_sha256 || 'Unavailable'}</dd></div></dl></details>
    </main>
    <footer className="border-t border-[#EEEEEE] bg-[#FAFAFA]"><div className="mx-auto flex max-w-[1320px] flex-wrap justify-between gap-3 px-5 py-6 text-[10px] text-[#858585] sm:px-8 lg:px-12"><span>Recurring Insights · Swiss AI Weeks 2026</span><span>Independent prototype for the UBS Transaction Activity Forecasting challenge.</span></div></footer>
  </div>;
}
