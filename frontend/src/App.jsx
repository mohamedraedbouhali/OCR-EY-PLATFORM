import { useCallback, useEffect, useRef, useState } from "react";
import Login from "./pages/Login.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import HistoryPage from "./pages/HistoryPage.jsx";
import TopBar from "./components/TopBar.jsx";
import AccountModal from "./components/AccountModal.jsx";
import { getToken, setToken, me } from "./api.js";

/** Minimal pathname-based router — no library needed for two pages. */
function useRoute() {
  const [pathname, setPathname] = useState(window.location.pathname);

  useEffect(() => {
    const onPopState = () => setPathname(window.location.pathname);
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = useCallback((to) => {
    window.history.pushState({}, "", to);
    setPathname(window.location.pathname);
  }, []);

  return [pathname, navigate];
}

export default function App() {
  const [user, setUser] = useState(null);
  const [checking, setChecking] = useState(Boolean(getToken()));
  const [showAccount, setShowAccount] = useState(false);
  const [toast, setToast] = useState(null);
  const toastTimer = useRef(null);
  const [pathname, navigate] = useRoute();

  const handleLogout = useCallback(() => {
    setToken(null);
    setUser(null);
  }, []);

  useEffect(() => {
    window.addEventListener("ey-logout", handleLogout);
    return () => window.removeEventListener("ey-logout", handleLogout);
  }, [handleLogout]);

  useEffect(() => {
    if (!getToken()) return;
    me()
      .then(setUser)
      .catch(() => setToken(null))
      .finally(() => setChecking(false));
  }, []);

  function notify(message) {
    setToast(message);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 2400);
  }

  if (checking) {
    return (
      <div className="splash">
        <img className="splash-logo" src="/ey.png" alt="EY" />
        <div className="splash-text">Loading…</div>
      </div>
    );
  }

  if (!user) {
    return <Login onLogin={setUser} />;
  }

  return (
    <div className="app-shell">
      <TopBar
        user={user}
        onBrandClick={() => navigate("/")}
        onAccountClick={() => setShowAccount(true)}
        onLogout={handleLogout}
      />

      {pathname === "/history" ? (
        <HistoryPage navigate={navigate} notify={notify} />
      ) : (
        <Dashboard user={user} navigate={navigate} notify={notify} />
      )}

      {showAccount && (
        <AccountModal
          user={user}
          onClose={() => setShowAccount(false)}
          onSaved={(fresh) => {
            setUser(fresh);
            setShowAccount(false);
            notify("Account updated");
          }}
        />
      )}

      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
    </div>
  );
}
