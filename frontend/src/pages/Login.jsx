import { useState } from "react";
import { login, setToken } from "../api.js";

const FEATURES = [
  {
    title: "GenAI vision extraction",
    text: "State-of-the-art recognition on degraded scans, tables and complex layouts.",
  },
  {
    title: "Multilingual by design",
    text: "French, English and Arabic — including mixed and right-to-left content.",
  },
  {
    title: "Secure document history",
    text: "Every extraction is stored per user and retrievable at any time.",
  },
];

export default function Login({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const res = await login(email.trim(), password);
      setToken(res.access_token);
      onLogin(res.user);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-page">
      {/* ---------- brand panel ---------- */}
      <aside className="login-brand">
        <div className="beam" aria-hidden="true" />
        <div className="beam beam-2" aria-hidden="true" />
        <img className="brand-logo" src="/ey.png" alt="EY" />
        <h1 className="brand-headline">
          Intelligent <span className="accent">OCR</span>
        </h1>
        <p className="brand-sub">
          Multilingual document intelligence powered by generative AI.
        </p>

        <ul className="feature-list">
          {FEATURES.map((f) => (
            <li key={f.title} className="feature">
              <span className="feature-tick" aria-hidden="true" />
              <div>
                <div className="feature-title">{f.title}</div>
                <div className="feature-text">{f.text}</div>
              </div>
            </li>
          ))}
        </ul>

        <p className="brand-footer">Building a better working world</p>
      </aside>

      {/* ---------- form panel ---------- */}
      <main className="login-main">
        <form onSubmit={handleSubmit} className="login-card" noValidate>
          <img className="login-card-logo" src="/ey.png" alt="" aria-hidden="true" />
          <h2 className="login-card-title">Sign in</h2>
          <p className="login-card-sub">Access your document workspace</p>

          <label className="field-label" htmlFor="email">
            Email
          </label>
          <input
            id="email"
            type="email"
            autoComplete="username"
            autoFocus
            placeholder="you@ey.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <label className="field-label" htmlFor="password">
            Password
          </label>
          <div className="password-wrap">
            <input
              id="password"
              type={showPassword ? "text" : "password"}
              autoComplete="current-password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <button
              type="button"
              className="password-toggle"
              onClick={() => setShowPassword((v) => !v)}
              aria-label={showPassword ? "Hide password" : "Show password"}
            >
              {showPassword ? "Hide" : "Show"}
            </button>
          </div>

          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}

          <button className="btn-primary login-submit" type="submit" disabled={busy}>
            {busy ? <span className="btn-spinner" aria-hidden="true" /> : null}
            {busy ? "Signing in…" : "Sign in"}
          </button>

          <p className="login-hint">
            Internal prototype — contact the administrator for access.
          </p>
        </form>
      </main>
    </div>
  );
}
