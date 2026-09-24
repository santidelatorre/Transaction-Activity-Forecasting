import { useEffect, useState } from 'react';
import { Activity, Database, FlaskConical, LayoutGrid, Menu, X } from 'lucide-react';
import { FOOTER_TEXT } from '@/lib/insights/constants';
import { Link } from '@/components/Link';

const NAV = [
    { href: '/insights/overview', label: 'Overview', icon: LayoutGrid },
    { href: '/insights/data', label: 'Submission', icon: Database },
    { href: '/insights/results', label: 'Technical results', icon: FlaskConical },
];

export default function InsightsLayout({ title, subtitle, children }) {
    const [open, setOpen] = useState(false);
    const [health, setHealth] = useState(null);

    useEffect(() => {
        document.title = title ? `${title} — Recurring Insights` : 'Recurring Insights';
    }, [title]);

    useEffect(() => {
        fetch('/api/v1/health')
            .then((response) => response.json())
            .then(setHealth)
            .catch(() => setHealth({ status: 'unavailable' }));
    }, []);

    const url = window.location.hash.slice(1).split('?')[0] || '/overview';
    const active = (href) => url === href.replace('/insights', '');

    return (
        <div className="min-h-screen bg-[#F8FAFC] text-[#0B2545] flex flex-col">
            <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 backdrop-blur">
                <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3 sm:px-6 lg:px-8">
                    <Link href="/insights/overview" className="flex items-center gap-2.5">
                        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#0F766E] text-white">
                            <Activity className="h-4 w-4" />
                        </span>
                        <span className="text-[15px] font-semibold tracking-tight">Recurring Insights</span>
                    </Link>

                    <nav className="ml-6 hidden items-center gap-1 md:flex">
                        {NAV.map((item) => (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                                    active(item.href)
                                        ? 'bg-teal-50 text-[#0F766E]'
                                        : 'text-slate-600 hover:bg-slate-100 hover:text-[#0B2545]'
                                }`}
                            >
                                {item.label}
                            </Link>
                        ))}
                    </nav>

                    <div className="ml-auto flex items-center gap-3">
                        <span className="hidden rounded-full bg-slate-50 px-3 py-1 text-xs font-semibold text-slate-600 ring-1 ring-slate-200 sm:inline-flex">
                            {health?.predictions_available ? 'Baseline output ready' : 'Waiting for baseline output'}
                        </span>
                        <button
                            type="button"
                            onClick={() => setOpen((v) => !v)}
                            aria-label="Toggle navigation"
                            className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 md:hidden"
                        >
                            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
                        </button>
                    </div>
                </div>

                {open && (
                    <nav className="border-t border-slate-200 bg-white px-4 py-2 md:hidden" onClick={() => setOpen(false)}>
                        {NAV.map((item) => (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium ${
                                    active(item.href) ? 'bg-teal-50 text-[#0F766E]' : 'text-slate-600'
                                }`}
                            >
                                <item.icon className="h-4 w-4" />
                                {item.label}
                            </Link>
                        ))}
                    </nav>
                )}
            </header>

            {!health?.predictions_available && (
                <div className="border-b border-amber-200 bg-amber-50/70">
                    <p className="mx-auto max-w-7xl px-4 py-2.5 text-xs text-amber-900 sm:px-6 lg:px-8">
                        Generate the baseline artifacts with <code>python scripts/run_ubs_baseline.py --config configs/ubs_v1.toml</code> to load real predictions.
                    </p>
                </div>
            )}

            <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6 lg:px-8">
                {title && (
                    <div className="mb-7">
                        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">{title}</h1>
                        {subtitle && <p className="mt-2 max-w-3xl text-sm text-slate-600">{subtitle}</p>}
                    </div>
                )}
                {children}
            </main>

            <footer className="border-t border-slate-200 bg-white">
                <div className="mx-auto max-w-7xl px-4 py-6 text-center text-xs text-slate-500 sm:px-6 lg:px-8">
                    <p>{FOOTER_TEXT}</p>
                    <p className="mt-2">
                        Made with ❤️ by{' '}
                        <a
                            href="https://laracopilot.com/"
                            target="_blank"
                            rel="noreferrer"
                            className="font-medium text-[#0F766E] hover:underline"
                        >
                            LaraCopilot
                        </a>
                    </p>
                </div>
            </footer>
        </div>
    );
}
