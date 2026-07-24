import { useEffect, useRef, useState } from "react";
import { updateProfile } from "../api.js";

/** Center-crop the picked image to a square and downscale to `size` px. */
function resizeToDataURI(file, size = 192) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const s = Math.min(img.width, img.height);
      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;
      canvas
        .getContext("2d")
        .drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, size, size);
      resolve(canvas.toDataURL("image/jpeg", 0.85));
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not read that image file"));
    };
    img.src = url;
  });
}

export default function AccountModal({ user, onClose, onSaved }) {
  const [name, setName] = useState(user.name || "");
  const [avatar, setAvatar] = useState(user.avatar || null);
  const [avatarChanged, setAvatarChanged] = useState(false);
  const [showPwd, setShowPwd] = useState(false);
  const [currentPwd, setCurrentPwd] = useState("");
  const [newPwd, setNewPwd] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const fileInput = useRef(null);

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function pickPhoto(file) {
    if (!file) return;
    setError("");
    try {
      setAvatar(await resizeToDataURI(file));
      setAvatarChanged(true);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleSave(e) {
    e.preventDefault();
    setError("");

    const payload = {};
    if (name.trim() && name.trim() !== user.name) payload.name = name.trim();
    if (avatarChanged) payload.avatar = avatar ?? "";
    if (newPwd) {
      payload.current_password = currentPwd;
      payload.new_password = newPwd;
    }
    if (Object.keys(payload).length === 0) {
      onClose();
      return;
    }

    setBusy(true);
    try {
      const fresh = await updateProfile(payload);
      onSaved(fresh);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <form
        className="modal-card"
        role="dialog"
        aria-modal="true"
        aria-label="Edit account"
        onClick={(e) => e.stopPropagation()}
        onSubmit={handleSave}
      >
        <div className="modal-head">
          <h2 className="modal-title">My account</h2>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        {/* ------- photo ------- */}
        <div className="avatar-editor">
          <button
            type="button"
            className="avatar-preview"
            onClick={() => fileInput.current?.click()}
            title="Change photo"
          >
            {avatar ? (
              <img src={avatar} alt="Profile" />
            ) : (
              <span>{(user.name || user.email)[0].toUpperCase()}</span>
            )}
            <span className="avatar-overlay">Edit</span>
          </button>
          <input
            ref={fileInput}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => pickPhoto(e.target.files[0])}
          />
          <div className="avatar-actions">
            <button type="button" className="btn-ghost" onClick={() => fileInput.current?.click()}>
              Upload photo
            </button>
            {avatar && (
              <button
                type="button"
                className="btn-ghost danger"
                onClick={() => {
                  setAvatar(null);
                  setAvatarChanged(true);
                }}
              >
                Remove
              </button>
            )}
          </div>
        </div>

        {/* ------- identity ------- */}
        <label className="field-label" htmlFor="acc-name">
          Display name
        </label>
        <input
          id="acc-name"
          type="text"
          value={name}
          maxLength={80}
          onChange={(e) => setName(e.target.value)}
          placeholder="Your name"
        />

        <label className="field-label">Email</label>
        <input type="email" value={user.email} disabled title="Email cannot be changed" />

        {/* ------- password ------- */}
        <button
          type="button"
          className="pwd-section-toggle"
          onClick={() => setShowPwd((v) => !v)}
        >
          {showPwd ? "▾" : "▸"} Change password
        </button>

        {showPwd && (
          <div className="pwd-section">
            <label className="field-label" htmlFor="acc-current">
              Current password
            </label>
            <input
              id="acc-current"
              type="password"
              autoComplete="current-password"
              value={currentPwd}
              onChange={(e) => setCurrentPwd(e.target.value)}
            />
            <label className="field-label" htmlFor="acc-new">
              New password
            </label>
            <input
              id="acc-new"
              type="password"
              autoComplete="new-password"
              value={newPwd}
              onChange={(e) => setNewPwd(e.target.value)}
            />
          </div>
        )}

        {error && (
          <div className="error-banner" role="alert">
            {error}
          </div>
        )}

        <div className="modal-actions">
          <button type="button" className="btn-ghost" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="btn-primary" disabled={busy}>
            {busy ? "Saving…" : "Save changes"}
          </button>
        </div>
      </form>
    </div>
  );
}
