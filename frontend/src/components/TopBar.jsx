export default function TopBar({ user, onBrandClick, onAccountClick, onLogout }) {
  return (
    <header className="topbar">
      <button className="topbar-brand" onClick={onBrandClick}>
        <img className="topbar-logo" src="/ey.png" alt="EY" />
        <div>
          <div className="topbar-title">Intelligent OCR</div>
          <div className="topbar-tag">Document Intelligence Platform</div>
        </div>
      </button>
      <div className="topbar-user">
        <button
          className="user-badge"
          onClick={onAccountClick}
          title="Edit account"
        >
          <span className="user-avatar">
            {user.avatar ? (
              <img src={user.avatar} alt="" />
            ) : (
              (user.name || user.email)[0].toUpperCase()
            )}
          </span>
          <span className="user-name">{user.name || user.email}</span>
          <span className="user-edit-hint" aria-hidden="true">
            ✎
          </span>
        </button>
        <button className="btn-ghost" onClick={onLogout}>
          Sign out
        </button>
      </div>
    </header>
  );
}
