import { useEffect, useId, useRef, useState } from 'react';
import { ArrowDown, ArrowRight, ArrowUpRight, ChevronDown, ChevronLeft, ChevronRight, RefreshCw } from 'lucide-react';

const RED = '#E60000';
const PAGE_SIZE = 6;
const format = (value, digits = 0) => typeof value === 'number' && Number.isFinite(value)
  ? value.toLocaleString('es-ES', { minimumFractionDigits: digits, maximumFractionDigits: digits, useGrouping: true })
  : '—';
const percent = (value) => typeof value === 'number' && Number.isFinite(value) ? `${format(value * 100, 1)} %` : '—';
const signed = (value) => typeof value === 'number' && Number.isFinite(value) ? `${value > 0 ? '+' : ''}${format(value, 4)}` : '—';
const families = { cloud: 'servicios en la nube', gym: 'gimnasio', insurance: 'seguros', mobile: 'telefonía móvil', music: 'música', software: 'software', streaming: 'streaming' };

function featureName(name) {
  let match = /^identity_(\w+)_(aliases|share|count)$/.exec(name);
  if (match) {
    const prefix = { aliases: 'Descripciones distintas asociadas a', share: 'Proporción de pagos con tarjeta asociados a', count: 'Pagos con tarjeta asociados a' }[match[2]];
    return `${prefix} ${families[match[1]] ?? match[1]}`;
  }
  match = /^mcc_(\d+)_(count|share)$/.exec(name);
  if (match) return `${match[2] === 'share' ? 'Proporción' : 'Número'} de operaciones · MCC ${match[1]}`;
  match = /^calendar_([A-Z]{3})_client_count_daily_mean_(\d+)d$/.exec(name);
  if (match) return `Media de operaciones diarias en ${match[1]} · últimos ${match[2]} días`;
  match = /^(transactions|outgoing|frequency)_last_(\d+)d$/.exec(name);
  if (match) return `${{ transactions: 'Operaciones', outgoing: 'Operaciones de salida', frequency: 'Frecuencia diaria de operaciones' }[match[1]]} · últimos ${match[2]} días`;
  return ({ n_transactions: 'Número de operaciones observadas', days_since_last_transaction: 'Días desde la última operación', history_days: 'Duración del historial', transactions_per_30d: 'Operaciones por cada 30 días', repeated_description_count: 'Descripciones que se repiten', best_recurrence_score: 'Mayor señal de recurrencia' })[name] ?? name;
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
        if (!response.ok) throw new Error(`No se pudieron cargar los resultados (${response.status}).`);
        return response.json();
      })
      .then((data) => setState({ data, loading: false, error: null }))
      .catch((error) => {
        if (error.name !== 'AbortError' || timedOut) setState({ data: null, loading: false, error: timedOut ? 'La consulta ha tardado demasiado. Inténtalo de nuevo.' : error.message });
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
  return <div role={error ? 'alert' : 'status'} className="flex min-h-48 flex-col items-center justify-center px-6 py-10 text-center">
    <span className={`mb-4 h-8 w-8 rounded-full border border-[#DADADA] border-t-[#E60000] ${loading ? 'motion-safe:animate-spin' : ''}`} aria-hidden="true" />
    <p className="max-w-sm text-sm leading-6 text-[#686868]">{loading ? 'Cargando los resultados del equipo…' : error || children}</p>
  </div>;
}

function MetricCard({ number, title, value, annotation, primary = false, children }) {
  return <article className={`relative flex min-h-[220px] flex-col rounded-sm border bg-white px-6 py-6 sm:px-7 ${primary ? 'border-[#E60000]' : 'border-[#E4E4E4]'}`}>
    <div className="flex min-h-10 items-start justify-between gap-3">
      <h3 className="max-w-[220px] text-sm font-medium leading-5 text-[#4D4D4D]">{title}</h3>
      <span className="font-mono text-[10px] text-[#999]">{number}</span>
    </div>
    <p className={`mt-5 text-[clamp(2.5rem,4.3vw,3.75rem)] font-medium leading-none tabular-nums tracking-[-0.055em] ${primary ? 'text-[#E60000]' : 'text-[#1A1A1A]'}`}>{value}</p>
    <p className="mt-4 text-xs leading-5 text-[#686868]">{annotation}</p>
    {children && <div className="mt-auto pt-4 text-[11px] text-[#686868]">{children}</div>}
  </article>;
}

function ProgressChart({ data, loading, error }) {
  const chartId = useId();
  const chartContainer = useRef(null);
  const [chartWidth, setChartWidth] = useState(760);
  const rows = data?.experiments ?? [];
  const selected = rows.findIndex((row) => row.result === 'selected');
  const focused = rows.findIndex((row) => row.candidate === 'focused');
  const [hovered, setHovered] = useState(null);
  const active = rows[hovered ?? selected] ?? rows[0];
  const width = chartWidth, height = 292, left = 40, right = 25, top = 28, bottom = 38;
  useEffect(() => {
    if (!chartContainer.current) return undefined;
    const observer = new ResizeObserver(([entry]) => setChartWidth(Math.max(280, Math.floor(entry.contentRect.width))));
    observer.observe(chartContainer.current);
    return () => observer.disconnect();
  }, [rows.length, loading, error]);
  const x = (index) => left + (index / Math.max(rows.length - 1, 1)) * (width - left - right);
  const y = (value) => top + (1 - value) * (height - top - bottom);
  const points = rows.map((row, i) => ({ x: x(i), y: y(row.metrics.macro_f1) }));
  const path = points.map((point, i) => {
    if (i === 0) return `M ${point.x} ${point.y}`;
    const previous = points[i - 1], middle = (previous.x + point.x) / 2;
    return `C ${middle} ${previous.y}, ${middle} ${point.y}, ${point.x} ${point.y}`;
  }).join(' ');

  return <section id="progress" className="scroll-mt-24 rounded-sm border border-[#E4E4E4] bg-white">
    <div className="px-6 pt-6 sm:px-7">
      <Eyebrow className="text-[#858585]">01 / Evolución del modelo</Eyebrow>
      <h2 className="mt-2 text-xl font-medium tracking-tight">Cada candidato, a la vista.</h2>
      <p className="mt-2 text-xs leading-5 text-[#686868]">Macro-F1 de los candidatos sobre el mismo conjunto VALID.</p>
    </div>
    {loading || error || !rows.length ? <Empty loading={loading} error={error}>El gráfico aparecerá cuando haya evaluaciones registradas. La meta aspiracional del equipo es 0,80.</Empty> : <>
      <div className="px-4 pt-4 sm:px-6">
        <div ref={chartContainer}>
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full overflow-visible" aria-labelledby={`${chartId}-title ${chartId}-desc`}>
          <title id={`${chartId}-title`}>Comparación de Macro-F1 con meta aspiracional de 0,80</title>
          <desc id={`${chartId}-desc`}>Los puntos siguen el orden del informe de validación, no el cronológico. V3-A es el modelo seleccionado. La anotación de corrección focalizada indica el uso de un umbral fijo de 0,85, sin búsqueda de umbral en VALID.</desc>
          {[0, 0.2, 0.4, 0.6, 0.8, 1].map((tick) => <g key={tick}>
            <line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} stroke={tick === 0.8 ? RED : '#EAEAEA'} strokeDasharray={tick === 0.8 ? '5 5' : '2 4'} strokeOpacity={tick === 0.8 ? 0.55 : 1} />
            <text x={left - 12} y={y(tick) + 4} textAnchor="end" fill="#888" fontSize="10">{format(tick, 1)}</text>
          </g>)}
          <text x={width - right} y={y(0.8) - 9} textAnchor="end" fill={RED} fontSize="10" letterSpacing="1">META ASPIRACIONAL · 0,80</text>
          <path d={`${path} L ${points.at(-1).x} ${y(0)} L ${left} ${y(0)} Z`} fill={RED} fillOpacity="0.035" />
          <path d={path} fill="none" stroke={RED} strokeWidth="2.4" />
          {rows.map((row, i) => <g key={row.experiment_id}>
            {i === selected && <circle cx={x(i)} cy={y(row.metrics.macro_f1)} r="10" fill={RED} fillOpacity="0.12" />}
            <circle cx={x(i)} cy={y(row.metrics.macro_f1)} r={i === selected ? 4.5 : 3.5} stroke={RED} strokeWidth="2" fill={i === selected ? RED : 'white'} />
            <circle cx={x(i)} cy={y(row.metrics.macro_f1)} r="14" fill="transparent" tabIndex="0" role="button" aria-label={`${row.model_name}, Macro-F1 ${format(row.metrics.macro_f1, 4)}`} onMouseEnter={() => setHovered(i)} onFocus={() => setHovered(i)} onClick={() => setHovered(i)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setHovered(i); } }} className="cursor-pointer outline-offset-4 focus-visible:outline focus-visible:outline-2 focus-visible:outline-[#1A1A1A]" />
            <text x={x(i)} y={height - 13} textAnchor="middle" fill={i === selected ? RED : '#888'} fontSize="10">E{row.sequence}</text>
          </g>)}
          {selected >= 0 && <g aria-hidden="true">
            <line x1={x(selected)} x2={x(selected)} y1={y(rows[selected].metrics.macro_f1) - 14} y2={y(rows[selected].metrics.macro_f1) - 29} stroke="#1A1A1A" strokeOpacity="0.25" />
            <text x={x(selected)} y={y(rows[selected].metrics.macro_f1) - 37} textAnchor="middle" fill="#1A1A1A" fontSize="10">Identidad por familia</text>
          </g>}
          {focused >= 0 && <g aria-hidden="true">
            <line x1={x(focused)} x2={x(focused)} y1={y(rows[focused].metrics.macro_f1) - 14} y2={y(rows[focused].metrics.macro_f1) - 29} stroke="#1A1A1A" strokeOpacity="0.25" />
            <text x={x(focused)} y={y(rows[focused].metrics.macro_f1) - 37} textAnchor="middle" fill="#1A1A1A" fontSize="10">Umbral fijo 0,85</text>
          </g>}
        </svg>
        </div>
      </div>
      <div className="mx-6 flex min-h-16 items-center justify-between gap-4 border-y border-[#EAEAEA] py-3 sm:mx-7" aria-live="polite">
        <div><p className="text-xs font-medium">{active?.model_name}</p><p className="mt-1 text-[11px] text-[#757575]">{active?.milestone}</p></div>
        <span className="text-xl tabular-nums tracking-tight">{format(active?.metrics.macro_f1, 4)}</span>
      </div>
      <p className="px-6 py-4 text-[11px] leading-5 text-[#757575] sm:px-7">Orden del informe, no cronología. VALID se reutilizó para comparar candidatos; esta curva no mide el rendimiento en TEST.</p>
    </>}
  </section>;
}

function ImportanceChart({ data, loading, error }) {
  const rows = data?.importance?.slice(0, 6) ?? [];
  const max = Math.max(1e-9, ...rows.map((row) => row.importance));
  return <section id="explainability" className="scroll-mt-24 rounded-sm border border-[#E4E4E4] bg-white">
    <div className="px-6 pt-6 sm:px-7">
      <Eyebrow className="text-[#858585]">02 / Señales detrás de la decisión</Eyebrow>
      <h2 className="mt-2 text-xl font-medium tracking-tight">¿En qué se fija el modelo?</h2>
      <p className="mt-2 text-xs leading-5 text-[#686868]">Importancia global medida en el componente CatBoost.</p>
    </div>
    {loading || error || !rows.length ? <Empty loading={loading} error={error}>Todavía no hay importancias medidas para este modelo.</Empty> : <>
      <div className="space-y-4 px-6 pb-4 pt-6 sm:px-7">
        {rows.map((row, index) => <div key={row.feature}>
          <div className="mb-2 flex items-start justify-between gap-3 text-xs leading-4">
            <span title={row.feature} className="max-w-[85%] text-[#4D4D4D]">{featureName(row.feature)}</span>
            <span className="shrink-0 tabular-nums text-[#686868]">{format(row.importance, 2)}</span>
          </div>
          <div className="h-1.5 bg-[#F0F0F0]" role="img" aria-label={`${featureName(row.feature)}: importancia ${format(row.importance, 3)}`}>
            <div className={`h-full ${index === 0 ? 'bg-[#E60000]' : 'bg-[#737373]'}`} style={{ width: `${row.importance / max * 100}%` }} />
          </div>
        </div>)}
      </div>
      <details className="mx-6 border-t border-[#EAEAEA] py-3 sm:mx-7">
        <summary className="cursor-pointer text-[11px] font-medium text-[#555]">Ver nombres originales y alcance</summary>
        <p className="mt-3 text-[11px] leading-5 text-[#686868]">{data.importance_scope}. Estas importancias globales no explican una predicción individual. La asociación de descripciones a familias es inferida.</p>
        <dl className="mt-3 space-y-2">{rows.map((row) => <div key={row.feature}><dt className="break-all font-mono text-[10px]">{row.feature}</dt><dd className="text-[11px] text-[#686868]">{featureName(row.feature)}</dd></div>)}</dl>
      </details>
      <p className="px-6 pb-5 text-[11px] leading-5 text-[#757575] sm:px-7">Escala propia de CatBoost. El modelo combina {percent(data.ensemble_weights?.catboost)} de CatBoost y {percent(data.ensemble_weights?.recurrence_heuristic)} de heurística de periodicidad.</p>
    </>}
  </section>;
}

function CandidateComparison({ data, loading, error }) {
  const rows = [...(data?.experiments ?? [])]
    .filter((row) => Number.isFinite(row.metrics?.macro_f1))
    .sort((left, right) => right.metrics.macro_f1 - left.metrics.macro_f1);
  return <section id="comparacion" className="mt-6 rounded-sm border border-[#E4E4E4] bg-white">
    <div className="px-6 pt-6 sm:px-7">
      <Eyebrow className="text-[#858585]">03 / Comparación experimental</Eyebrow>
      <h2 className="mt-2 text-xl font-medium tracking-tight">Qué aportó cada candidato.</h2>
      <p className="mt-2 text-xs leading-5 text-[#686868]">Macro-F1 medido en VALID. Se muestran las variantes registradas en el informe.</p>
    </div>
    {loading || error || !rows.length ? <Empty loading={loading} error={error}>La comparación aparecerá cuando estén disponibles las evaluaciones.</Empty> : <>
      <div className="space-y-4 px-6 py-6 sm:px-7">
        {rows.map((row) => <div key={row.experiment_id}>
          <div className="mb-2 flex items-baseline justify-between gap-4 text-xs">
            <span className="font-medium text-[#333]">{row.model_name}{row.result === 'selected' && <span className="ml-2 text-[10px] font-semibold uppercase tracking-wide text-[#C90000]">Modelo seleccionado</span>}</span>
            <span className="shrink-0 font-mono tabular-nums text-[#555]">{format(row.metrics.macro_f1, 4)}</span>
          </div>
          <div className="h-2 bg-[#F0F0F0]" role="img" aria-label={`${row.model_name}: Macro-F1 ${format(row.metrics.macro_f1, 4)} en VALID`}>
            <div className={`h-full ${row.result === 'selected' ? 'bg-[#E60000]' : 'bg-[#737373]'}`} style={{ width: `${Math.max(0, Math.min(100, row.metrics.macro_f1 * 100))}%` }} />
          </div>
        </div>)}
      </div>
      <p className="mx-6 border-t border-[#EAEAEA] py-4 text-[11px] leading-5 text-[#686868] sm:mx-7">
        Composición del modelo V3-A seleccionado: {percent(data?.ensemble_weights?.catboost)} CatBoost + {percent(data?.ensemble_weights?.recurrence_heuristic)} heurística de periodicidad. Los pesos describen la mezcla, no son scores de los componentes.
      </p>
      <p className="px-6 pb-5 text-[11px] leading-5 text-[#757575] sm:px-7">VALID se reutilizó para comparar candidatos; estas métricas no estiman el rendimiento en TEST.</p>
    </>}
  </section>;
}

function ExperimentTable({ revision, ensembleWeights }) {
  const [page, setPage] = useState(0);
  const [expanded, setExpanded] = useState(null);
  const { data, loading, error } = useResource(`/api/v1/experiments?limit=${PAGE_SIZE + 1}&offset=${page * PAGE_SIZE}`, revision);
  const rows = data?.experiments?.slice(0, PAGE_SIZE) ?? [];
  const hasNext = (data?.experiments?.length ?? 0) > PAGE_SIZE;
  return <section id="experiments" className="mt-6 scroll-mt-24 rounded-sm border border-[#E4E4E4] bg-white">
    <div className="flex flex-wrap items-end justify-between gap-4 px-6 py-6 sm:px-7">
      <div><Eyebrow className="text-[#858585]">04 / Resultados verificables</Eyebrow><h2 className="mt-2 text-xl font-medium tracking-tight">La evidencia detrás de cada decisión.</h2></div>
      <span className="border border-[#E4E4E4] px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-[#686868]">Historial de evaluación</span>
    </div>
    {loading || error || !rows.length ? <Empty loading={loading} error={error}>No hay experimentos en esta página. Los resultados aparecerán al cargar el registro del equipo.</Empty> : <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-left text-xs">
        <caption className="sr-only">Experimentos reales. Macro-F1 y accuracy en cada conjunto de evaluación. Selecciona una fila para consultar su procedencia.</caption>
        <thead className="border-y border-[#E4E4E4] bg-[#F5F5F5] text-[10px] font-medium uppercase tracking-[0.09em] text-[#757575]">
          <tr><th scope="col" className="px-6 py-3.5 sm:pl-7">Experimento / modelo</th><th scope="col" className="px-4 py-3.5">Evaluación</th><th scope="col" className="px-4 py-3.5 text-right">Macro-F1</th><th scope="col" className="px-4 py-3.5 text-right">Accuracy</th><th scope="col" className="px-4 py-3.5 text-right">Δ vs. baseline</th><th scope="col" className="px-4 py-3.5">Pesos / mezcla</th><th scope="col" className="px-6 py-3.5 text-right sm:pr-7">Estado</th></tr>
        </thead>
        <tbody>{rows.map((row, index) => <ExperimentRows key={row.experiment_id} row={row} ensembleWeights={ensembleWeights} index={index} expanded={expanded === row.experiment_id} toggle={() => setExpanded(expanded === row.experiment_id ? null : row.experiment_id)} />)}</tbody>
      </table>
    </div>}
    <div className="flex items-center justify-between gap-4 border-t border-[#E4E4E4] px-6 py-4 text-[11px] text-[#686868] sm:px-7">
      <span>Página {page + 1}<span className="hidden sm:inline"> · Δ respecto a la baseline de cada registro</span></span>
      <div className="flex gap-2">
        <button type="button" aria-label="Página anterior de experimentos" disabled={page === 0 || loading} onClick={() => { setPage(page - 1); setExpanded(null); }} className="flex items-center gap-1 border border-[#DADADA] px-3 py-2 hover:bg-[#F5F5F5] disabled:cursor-not-allowed disabled:opacity-35"><ChevronLeft size={13} /> Anterior</button>
        <button type="button" aria-label="Página siguiente de experimentos" disabled={!hasNext || loading} onClick={() => { setPage(page + 1); setExpanded(null); }} className="flex items-center gap-1 border border-[#DADADA] px-3 py-2 hover:bg-[#F5F5F5] disabled:cursor-not-allowed disabled:opacity-35">Siguiente <ChevronRight size={13} /></button>
      </div>
    </div>
  </section>;
}

function ExperimentRows({ row, ensembleWeights, index, expanded, toggle }) {
  const scope = row.metrics?.scope;
  const mix = row.candidate === 'A' && ensembleWeights
    ? `${percent(ensembleWeights.catboost)} CatBoost + ${percent(ensembleWeights.recurrence_heuristic)} heurística`
    : row.candidate === 'ensemble' ? '50 % V3 + 50 % V2' : '—';
  return <>
    <tr className={`border-t border-[#EEEEEE] ${index % 2 ? 'bg-[#FAFAFA]' : 'bg-white'}`}>
      <th scope="row" className="px-6 py-4 font-normal sm:pl-7">
        <button type="button" onClick={toggle} aria-expanded={expanded} className="flex items-center gap-2 text-left hover:text-[#E60000]">
          <ChevronDown size={13} className={`shrink-0 transition-transform ${expanded ? 'rotate-180' : ''}`} />
          <span><span className="block font-medium">{row.model_name}</span><span className="mt-1 block font-mono text-[10px] text-[#888]">{row.sequence ? `E${row.sequence}` : row.experiment_id?.slice(0, 8)} · {row.model_version}</span></span>
        </button>
      </th>
      <td className="px-4 py-4 text-[#686868]">{scope === 'validation' ? 'VALID' : scope || 'Sin especificar'}</td>
      <td className="px-4 py-4 text-right font-medium tabular-nums">{format(row.metrics?.macro_f1, 4)}</td>
      <td className="px-4 py-4 text-right tabular-nums text-[#686868]">{percent(row.metrics?.accuracy)}</td>
      <td className="px-4 py-4 text-right tabular-nums text-[#686868]">{signed(row.delta_vs_baseline)}</td>
      <td className="px-4 py-4 text-[11px] text-[#686868]">{mix}</td>
      <td className="px-6 py-4 text-right sm:pr-7"><span className={`inline-block whitespace-nowrap px-2 py-1 text-[10px] ${row.result === 'selected' ? 'border border-[#FFD1D1] bg-[#FFF6F6] text-[#C90000]' : 'text-[#757575]'}`}>{row.result === 'selected' ? 'Modelo seleccionado' : 'Evaluado'}</span></td>
    </tr>
    {expanded && <tr className="border-t border-[#EEEEEE] bg-[#F7F7F7]"><td colSpan={7} className="px-7 py-5 text-xs leading-6 text-[#686868]">
      <p>{row.notes || 'Sin anotaciones adicionales en el registro.'}</p>
      <p>Clientes evaluados: {format(row.metrics?.validation_clients)} · Fuente: {row.source || 'Registro de experimentos'}.</p>
      <p className="break-all font-mono text-[10px]">Commit de evaluación: {row.git_commit || 'No registrado'} · Run: {row.metrics?.run_id || 'No registrado'}</p>
    </td></tr>}
  </>;
}

export default function App() {
  const [revision, setRevision] = useState(0);
  const { data, loading, error } = useResource('/api/v1/dashboard', revision);
  const metrics = data?.metrics;
  const ready = Boolean(data?.available);
  useEffect(() => { document.title = 'Recurring Insights · Una visión de lo que viene'; document.documentElement.lang = 'es'; }, []);

  return <div className="min-h-screen bg-white font-sans text-[#1A1A1A] selection:bg-[#FFE2E2]">
    <a href="#content" className="sr-only z-50 bg-white p-4 focus:not-sr-only focus:fixed">Saltar al contenido</a>
    <header className="sticky top-0 z-30 border-b border-[#EAEAEA] bg-white/95 backdrop-blur-sm">
      <div className="mx-auto flex h-[76px] max-w-[1320px] items-center justify-between gap-5 px-5 sm:px-8 lg:px-12">
        <a href="#content" className="flex shrink-0 items-center gap-3" aria-label="Recurring Insights, inicio">
          <span className="grid h-8 w-8 grid-cols-2 items-end gap-[3px] border-b-[3px] border-[#E60000] pb-[3px]" aria-hidden="true"><span className="h-3 bg-[#E60000]" /><span className="h-6 bg-[#E60000]" /></span>
          <span className="text-base font-semibold tracking-[-0.035em]">Recurring Insights<span className="mt-0.5 block text-[9px] font-medium uppercase tracking-[0.22em] text-[#888]">Transaction intelligence</span></span>
        </a>
        <nav aria-label="Navegación principal" className="hidden items-center gap-7 text-xs text-[#686868] md:flex">
          <a href="#quality" className="hover:text-[#E60000]">Visión general</a><a href="#explainability" className="hover:text-[#E60000]">Cómo decide</a><a href="#comparacion" className="hover:text-[#E60000]">Modelos</a><a href="#experiments" className="hover:text-[#E60000]">Experimentos</a>
        </nav>
        <button type="button" onClick={() => setRevision((value) => value + 1)} disabled={loading} className="flex items-center gap-2 border border-[#DADADA] px-3 py-2 text-[11px] hover:border-[#1A1A1A] disabled:opacity-50" aria-label="Actualizar resultados"><RefreshCw size={12} className={loading ? 'motion-safe:animate-spin' : ''} /><span className="hidden sm:inline">Actualizar</span></button>
      </div>
    </header>

    <main id="content" className="mx-auto max-w-[1320px] scroll-mt-24 px-5 pb-14 sm:px-8 lg:px-12">
      <section className="grid gap-10 pb-10 pt-10 sm:pt-14 lg:grid-cols-[1fr_260px] lg:items-center lg:gap-20">
        <div>
          <Eyebrow className="text-[#E60000]">UBS challenge / Swiss AI Weeks</Eyebrow>
          <h1 className="mt-5 max-w-[780px] text-[clamp(2.25rem,4.2vw,3.5rem)] font-medium leading-[1.1] tracking-[-0.045em]">Cada movimiento cuenta.<br /><span className="text-[#777]">Entendamos qué viene después.</span></h1>
          <p className="mt-5 max-w-[600px] text-sm leading-7 text-[#686868]">Convertimos el historial de transacciones en una señal sobre el próximo compromiso recurrente. Una base para conversaciones financieras mejor informadas.</p>
          <a href="#explainability" className="mt-6 inline-flex items-center gap-3 text-xs font-semibold text-[#E60000]">Explorar la evidencia <ArrowDown size={14} /></a>
        </div>
        <aside className="relative overflow-hidden bg-[#1A1A1A] px-7 py-6 text-white lg:py-8" aria-label="Horizonte de predicción">
          <Eyebrow className="text-[#B8B8B8]">La mirada hacia delante</Eyebrow>
          <p className="mt-5 text-6xl font-light leading-none tracking-[-0.055em]">90<span className="ml-2 text-base font-normal tracking-normal text-[#B8B8B8]">días</span></p>
          <div className="my-5 flex items-center gap-2" aria-hidden="true"><span className="h-1.5 w-1.5 rounded-full bg-[#E60000]" /><span className="h-px flex-1 bg-[#666]" /><ArrowRight size={13} className="text-[#E60000]" /></div>
          <p className="text-[11px] leading-5 text-[#B8B8B8]">Una familia recurrente por cliente.<br />Corte del historial: 1 ene. 2026.</p>
        </aside>
      </section>

      <section id="quality" className="scroll-mt-24 border-t border-[#EAEAEA] pt-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <Eyebrow className="text-[#757575]">Resultados medidos</Eyebrow>
          <span className="flex items-center gap-2 text-[11px] text-[#686868]"><span className={`h-1.5 w-1.5 rounded-full ${ready ? 'bg-[#1A1A1A]' : 'bg-[#B8B8B8]'}`} />{loading ? 'Cargando evaluación' : ready ? `${data.model_version} · evaluación VALID` : 'Evaluación pendiente de cargar'}</span>
        </div>
        {error && <div role="alert" className="mb-4 border-l-2 border-[#E60000] bg-[#FFF6F6] px-4 py-3 text-sm text-[#8A2020]">{error} Usa «Actualizar» para volver a consultar.</div>}
        {!loading && !error && !ready && <p role="status" className="mb-4 border-l-2 border-[#DADADA] bg-[#F5F5F5] px-4 py-3 text-sm text-[#686868]">Aún no se han cargado los resultados de V3-A en este entorno. Las métricas aparecerán al conectar los artefactos de evaluación.</p>}
        <div className="grid gap-4 md:grid-cols-3">
          <MetricCard primary number="01" title="Calidad de predicción general" value={format(metrics?.macro_f1, 4)} annotation="Macro-F1 · las ocho clases tienen el mismo peso.">
            <span className="inline-flex items-center gap-1 text-[#1A1A1A]">{metrics?.delta_vs_baseline != null && <ArrowUpRight size={12} />}{metrics?.delta_vs_baseline != null ? `${signed(metrics.delta_vs_baseline)} puntos F1 frente a V2` : 'Sin resultado medido disponible'}</span>
            <span className="mt-1 block">Meta aspiracional del equipo: ≥ 0,80 F1 · no es un resultado alcanzado.</span>
          </MetricCard>
          <MetricCard number="02" title="Tasa media de detección" value={percent(metrics?.macro_recall)} annotation="Recall macro · media del recall de las ocho clases, con igual peso.">
            <span>Accuracy global: {percent(metrics?.accuracy)} · medido en VALID.</span>
          </MetricCard>
          <MetricCard number="03" title="Clientes evaluados" value={format(metrics?.validation_clients)} annotation="Clientes de validación contados a partir de la matriz de confusión.">
            <span>7 familias recurrentes + la categoría «none».</span>
          </MetricCard>
        </div>
      </section>

      <div className="mt-6 grid items-stretch gap-6 lg:grid-cols-[1.15fr_1fr]">
        <ProgressChart data={data} loading={loading} error={error} />
        <ImportanceChart data={data} loading={loading} error={error} />
      </div>
      <CandidateComparison data={data} loading={loading} error={error} />
      <ExperimentTable revision={revision} ensembleWeights={data?.ensemble_weights} />

      <section id="method" className="mt-10 grid gap-7 border-y border-[#E4E4E4] py-8 md:grid-cols-[1.1fr_1fr_1fr]">
        <div><Eyebrow className="text-[#E60000]">La confianza también se explica</Eyebrow><h2 className="mt-3 max-w-xs text-xl font-medium leading-7 tracking-tight">Una señal útil.<br />Con sus límites a la vista.</h2></div>
        <div><h3 className="text-xs font-semibold">La historia precede a la predicción</h3><p className="mt-2 text-xs leading-6 text-[#686868]">El modelo utiliza transacciones anteriores al corte. Las descripciones pueden ser ambiguas; las asociaciones a familias se aprenden con separación por cliente.</p></div>
        <div><h3 className="text-xs font-semibold">La validación tiene un alcance</h3><p className="mt-2 text-xs leading-6 text-[#686868]">VALID se reutilizó para comparar candidatos. «None» significa que no se predice una familia recurrente en el horizonte; no garantiza la ausencia de movimientos.</p></div>
      </section>
      <details className="group border-b border-[#E4E4E4] py-4 text-xs text-[#686868]">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-3"><span>Ficha del modelo y trazabilidad</span><ChevronDown size={14} className="group-open:rotate-180" /></summary>
        <dl className="mt-4 grid gap-4 text-[11px] sm:grid-cols-2">
          <div><dt className="font-medium text-[#1A1A1A]">Modelo seleccionado</dt><dd className="mt-1">{data?.model_version || 'Pendiente'} · horizonte de 90 días</dd></div>
          <div><dt className="font-medium text-[#1A1A1A]">Commit de la baseline</dt><dd className="mt-1 break-all font-mono text-[10px]">{data?.base_sha || 'No disponible'}</dd></div>
          <div><dt className="font-medium text-[#1A1A1A]">Commit del informe de evaluación</dt><dd className="mt-1 break-all font-mono text-[10px]">{data?.evaluation_sha || 'No registrado'}</dd></div>
          <div><dt className="font-medium text-[#1A1A1A]">Huella del informe (SHA-256)</dt><dd className="mt-1 break-all font-mono text-[10px]">{data?.report_sha256 || 'No disponible'}</dd></div>
        </dl>
      </details>
    </main>
    <footer className="border-t border-[#EEEEEE] bg-[#FAFAFA]"><div className="mx-auto flex max-w-[1320px] flex-wrap justify-between gap-3 px-5 py-6 text-[10px] text-[#858585] sm:px-8 lg:px-12"><span>Recurring Insights · Swiss AI Weeks 2026</span><span>Prototipo independiente para el reto UBS Transaction Activity Forecasting.</span></div></footer>
  </div>;
}
