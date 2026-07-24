const TOKEN_KEY = "ey_ocr_token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(`/api${path}`, { ...options, headers });

  if (res.status === 401) {
    setToken(null);
    window.dispatchEvent(new Event("ey-logout"));
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  return res.json();
}

export function login(email, password) {
  return request("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
}

export function me() {
  return request("/auth/me");
}

export function meta() {
  return request("/meta");
}

export function updateProfile(payload) {
  return request("/auth/profile", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function runOCR(file, mode, langs, engine) {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", mode);
  if (engine) form.append("engine", engine);
  form.append("langs", langs.join(","));
  return request("/ocr", { method: "POST", body: form });
}

export function history() {
  return request("/history");
}

export function getDocument(id) {
  return request(`/history/${id}`);
}

export function deleteDocument(id) {
  return request(`/history/${id}`, { method: "DELETE" });
}
