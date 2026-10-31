import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { ApiError, artifactUrl } from '../../../shared/api/client';
import { listInspections } from '../api/component1Api';
import type { InspectionListItem } from '../types/component1';

import InspectionLayout from '../components/InspectionLayout';

const CLASS_NAMES = ['baekra', 'color issues', 'contamination', 'cut', 'gray stitch', 'selvet', 'stain'];
const PAGE_SIZE = 10;

export default function HistoryPage() {
  const [items, setItems] = useState<InspectionListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [classFilter, setClassFilter] = useState('');
  const [fromDate, setFromDate] = useState('');
  const [toDate, setToDate] = useState('');
  const reqRef = useRef(0);

  const fetchPage = useCallback(
    async (p: number) => {
      const rid = ++reqRef.current;
      setLoading(true);
      setError('');
      try {
        const data = await listInspections({
          skip: p * PAGE_SIZE,
          limit: PAGE_SIZE,
          class_name: classFilter || undefined,
          from_date: fromDate ? new Date(fromDate).toISOString() : undefined,
          to_date: toDate ? new Date(toDate + 'T23:59:59Z').toISOString() : undefined,
        });
        if (rid !== reqRef.current) return;
        setItems(data.items);
        setTotal(data.total);
      } catch (err) {
        if (rid !== reqRef.current) return;
        setError(err instanceof ApiError ? err.detail : 'Failed to load inspections.');
      } finally {
        if (rid === reqRef.current) setLoading(false);
      }
    },
    [classFilter, fromDate, toDate],
  );

  useEffect(() => {
    void fetchPage(page);
    return () => { ++reqRef.current; };
  }, [page, fetchPage]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return <InspectionLayout title="Inspection History" description="Revisit inspections, review detected classes and open saved explanations.">
    <section className="panel"><div className="toolbar"><label>Class<select value={classFilter} onChange={e => { setClassFilter(e.target.value); setPage(0); }}><option value="">All classes</option>{CLASS_NAMES.map(c => <option key={c}>{c}</option>)}</select></label><label>From (UTC)<input type="date" value={fromDate} onChange={e => { setFromDate(e.target.value); setPage(0); }} /></label><label>To (UTC)<input type="date" value={toDate} min={fromDate || undefined} onChange={e => { setToDate(e.target.value); setPage(0); }} /></label><button className="secondary" onClick={() => { setClassFilter(''); setFromDate(''); setToDate(''); setPage(0); }}>Clear filters</button></div></section>
    {error ? <div className="panel" style={{ marginTop:24 }}><p role="alert">{error}</p><button onClick={() => void fetchPage(page)}>Retry</button></div> : loading ? <p role="status" className="panel">Loading inspections…</p> : items.length === 0 ? <section className="panel" style={{ marginTop:24 }}><h2>No inspections found.</h2><p className="muted">Try different filters or start a new fabric inspection.</p><Link className="button" to="/component1">New Inspection</Link></section> : <>
      <p className="muted" role="status">{total} inspection{total !== 1 && 's'} · Page {page + 1} of {totalPages}</p>
      <div className="history-list">{items.map(item => <article key={item.id} className="panel history-item">{item.thumbnail_url ? <img className="thumbnail" loading="lazy" src={artifactUrl(item.thumbnail_url)} alt="Fabric thumbnail" /> : <div className="thumbnail" aria-label="No thumbnail" />}<div><Link to={`/component1/inspections/${item.id}`}>{new Date(item.detected_at).toLocaleString()}</Link><div className="muted file-meta">{item.model_version || 'Model not recorded'}</div></div><div><strong>{item.total_defects} defects</strong><div className="muted">{item.defect_classes.join(', ') || 'No detected classes'}</div></div><Link aria-label={`View inspection from ${new Date(item.detected_at).toLocaleString()}`} to={`/component1/inspections/${item.id}`}>View details →</Link></article>)}</div>
      <nav className="toolbar" aria-label="History pagination" style={{ marginTop:24 }}><button className="secondary" disabled={page === 0} onClick={() => setPage(p => p - 1)}>← Prev</button><span>{page + 1} / {totalPages}</span><button className="secondary" disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}>Next →</button></nav>
    </>}
  </InspectionLayout>;
}
