import { useMemo, useState } from "react";
import useHistoryList from "../hooks/useHistoryList.js";

const MODE_LABELS = {
  genai_vision: "VLM Vision",
  hybrid: "Hybrid (Tesseract + VLM)",
  tesseract_only: "Tesseract only",
};

function fmtDate(iso) {
  return new Date(iso).toLocaleString(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function HistoryPage({ navigate, notify }) {
  const { items, remove } = useHistoryList();
  const [query, setQuery] = useState("");
  const list = items || [];

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return list;
    return list.filter(
      (i) =>
        i.filename.toLowerCase().includes(q) ||
        (MODE_LABELS[i.mode] || i.mode).toLowerCase().includes(q) ||
        (i.preview || "").toLowerCase().includes(q)
    );
  }, [list, query]);

  function open(id) {
    navigate(`/?doc=${id}`);
  }

  async function handleDelete(id, e) {
    e.preventDefault();
    e.stopPropagation();
    try {
      await remove(id);
      notify("Document deleted");
    } catch (err) {
      notify(err.message);
    }
  }

  return (
    <main className="history-page">
      <div className="history-page-head">
        <div>
          <h1 className="history-page-title">
            Historique
            {list.length > 0 && <span className="count-badge">{list.length}</span>}
          </h1>
          <p className="welcome-sub">
            Every extraction you've run, searchable and kept permanently.
          </p>
        </div>
        <div className="history-page-actions">
          {list.length > 0 && (
            <input
              type="search"
              className="history-search wide"
              placeholder="Search documents…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              autoFocus
            />
          )}
          <button className="btn-primary" onClick={() => navigate("/")}>
            + New extraction
          </button>
        </div>
      </div>

      <div className="history-page-body">
        {items === null && (
          <div className="history-grid">
            {[0, 1, 2, 3].map((i) => (
              <div
                key={i}
                className="skeleton"
                style={{ animationDelay: `${i * 100}ms` }}
              />
            ))}
          </div>
        )}

        {items !== null && list.length === 0 && (
          <div className="result-empty">
            <img className="empty-logo" src="/ey.png" alt="" aria-hidden="true" />
            <p className="empty-title">No documents yet</p>
            <p className="muted-note">
              Every extraction you run is saved here permanently.
            </p>
            <button className="btn-primary" onClick={() => navigate("/")}>
              Run your first OCR
            </button>
          </div>
        )}

        {list.length > 0 && visible.length === 0 && (
          <p className="muted-note">No match for “{query}”.</p>
        )}

        {visible.length > 0 && (
          <div className="history-grid">
            {visible.map((it, i) => (
              <div
                key={it.id}
                className="history-item card"
                style={{ animationDelay: `${Math.min(i, 10) * 30}ms` }}
              >
                <a
                  href={`/?doc=${it.id}`}
                  className="history-link"
                  onClick={(e) => {
                    e.preventDefault();
                    open(it.id);
                  }}
                >
                  <div className="history-name">{it.filename}</div>
                  <div className="history-meta">
                    {MODE_LABELS[it.mode] || it.mode} · {it.duration_s.toFixed(1)}s ·{" "}
                    {fmtDate(it.created_at)}
                  </div>
                  {it.preview && (
                    <div className="history-preview">{it.preview}</div>
                  )}
                </a>
                <button
                  className="history-delete"
                  title="Delete"
                  onClick={(e) => handleDelete(it.id, e)}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
