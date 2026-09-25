import { useEffect, useState } from 'react';
import ClientDetail from '@/pages/ClientDetail';
import ImportExport from '@/pages/ImportExport';
import ExecutiveDashboard from '@/pages/ExecutiveDashboard';
import TechnicalResults from '@/pages/TechnicalResults';

export default function App() {
  const [hash, setHash] = useState(() => window.location.hash);
  const route = hash.slice(1).split('?')[0] || '/overview';

  useEffect(() => {
    const update = () => setHash(window.location.hash);
    window.addEventListener('hashchange', update);
    return () => window.removeEventListener('hashchange', update);
  }, []);

  if (route === '/client') return <ClientDetail key={hash} />;
  if (route === '/data') return <ImportExport />;
  if (route === '/results') return <TechnicalResults />;
  return <ExecutiveDashboard />;
}
