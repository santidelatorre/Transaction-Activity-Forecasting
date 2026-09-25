import { useEffect, useState } from 'react';
import { Database, FlaskConical, LayoutGrid, Menu, X } from 'lucide-react';
import { FOOTER_TEXT } from '@/lib/insights/constants';
import { Link } from '@/components/Link';

const NAV = [
    { href: '/insights/overview', label: 'Dashboard', icon: LayoutGrid },
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
        <div className="flex min-h-screen flex-col bg-white font-sans text-[#1A1A1A]">
            <header className="sticky top-0 z-30 border-b border-[#EBEBEB] bg-white">
                <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3 sm:px-6 lg:px-8">
                    <Link href="/insights/overview" className="flex items-center gap-2.5">
                        <span className="flex h-8 w-8 items-center justify-center bg-[#E60000] text-xs font-bold tracking-tight text-white">RI</span>
                        <span className="text-[15px] font-semibold tracking-tight">Recurring Insights</span>
                        <span className="hidden border-l border-[#D6D6D6] pl-3 text-xs text-[#666] sm:inline">Hackathon prototype</span>
                    </Link>

                    <nav className="ml-6 hidden items-center gap-1 md:flex">
                        {NAV.map((item) => (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={`border-b-2 px-3 py-2 text-sm font-medium transition-colors ${
                                    active(item.href)
                                        ? 'border-[#E60000] text-[#1A1A1A]'
                                        : 'border-transparent text-[#666] hover:text-[#E60000]'
                                }`}
                            >
                                {item.label}
                            </Link>
                        ))}
                    </nav>

                    <div className="ml-auto flex items-center gap-3">
                        <span className="hidden border border-[#EBEBEB] px-3 py-1 text-xs font-medium text-[#666] sm:inline-flex">
                            {health?.predictions_available ? 'Baseline output ready' : 'Waiting for baseline output'}
                        </span>
                        <button
                            type="button"
                            onClick={() => setOpen((v) => !v)}
                            aria-label="Toggle navigation"
                            className="p-2 text-[#666] hover:bg-[#F5F5F5] md:hidden"
                        >
                            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
                        </button>
                    </div>
                </div>

                {open && (
                    <nav className="border-t border-[#EBEBEB] bg-white px-4 py-2 md:hidden" onClick={() => setOpen(false)}>
                        {NAV.map((item) => (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={`flex items-center gap-3 px-3 py-2.5 text-sm font-medium ${
                                    active(item.href) ? 'border-l-2 border-[#E60000] text-[#E60000]' : 'text-[#666]'
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
                <div className="border-b border-[#EBEBEB] bg-[#F5F5F5]">
                    <p className="mx-auto max-w-7xl px-4 py-2.5 text-xs text-[#555] sm:px-6 lg:px-8">
                        Generate the baseline artifacts with <code>python scripts/run_ubs_baseline.py --config configs/ubs_v1.toml</code> to load real predictions.
                    </p>
                </div>
            )}

            <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6 lg:px-8">
                {title && (
                    <div className="mb-7">
                        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h1>
                        {subtitle && <p className="mt-2 max-w-3xl text-sm text-[#666]">{subtitle}</p>}
                    </div>
                )}
                {children}
            </main>

            <footer className="border-t border-[#EBEBEB] bg-white">
                <div className="mx-auto max-w-7xl px-4 py-6 text-center text-xs text-[#666] sm:px-6 lg:px-8">
                    <p>{FOOTER_TEXT}</p>
                    <p className="mt-2">
                        Made with ❤️ by{' '}
                        <a
                            href="https://laracopilot.com/"
                            target="_blank"
                            rel="noreferrer"
                            className="font-medium text-[#E60000] hover:underline"
                        >
                            LaraCopilot
                        </a>
                    </p>
                </div>
            </footer>
        </div>
    );
}
