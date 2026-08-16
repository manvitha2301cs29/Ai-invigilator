// api/client.js
// -------------
// A thin fetch wrapper around the Phase G backend. Reads the deployed
// backend URL from VITE_API_URL (docs/03_DEPLOYMENT_GUIDE.txt Part 1
// Step 4: "set an environment variable pointing the frontend at your
// deployed backend URL"), defaulting to localhost for dev.
//
// The JWT (from POST /auth/login) is kept in memory + localStorage
// under one key -- this dashboard never touches the watcher_token; that
// credential is shown to the student once (see SignupPage) so they can
// paste it into their local watcher's config, and is never stored or
// used by this frontend again.

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const TOKEN_KEY = "invigilator_dashboard_jwt";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

async function request(path, { method = "GET", body, auth = true } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (auth) {
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
  }
  const res = await fetch(`${API_URL}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch {
      // response wasn't JSON -- keep statusText
    }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  if (res.status === 204 || res.status === 202) return null;
  return res.json();
}

export const api = {
  // Auth / account
  signUp: (name, email, password) =>
    request("/users", { method: "POST", body: { name, email, password }, auth: false }),
  login: (email, password) =>
    request("/auth/login", { method: "POST", body: { email, password }, auth: false }),

  // Planner
  listUpcomingBlocks: () => request("/dashboard/target-blocks"),
  createBlock: (plannedStart, plannedEnd, plannedDurationMin) =>
    request("/dashboard/target-blocks", {
      method: "POST",
      body: {
        planned_start: plannedStart,
        planned_end: plannedEnd,
        planned_duration_min: plannedDurationMin,
      },
    }),

  // Recommendation / summary
  getRecommendation: (blockId, tone) =>
    request(`/target-blocks/${blockId}/recommendation?tone=${tone}`),

  // History
  getHistory: (limit = 20) => request(`/target-blocks/history?limit=${limit}`),
};

export function wsUrl(path) {
  const httpUrl = new URL(API_URL);
  const scheme = httpUrl.protocol === "https:" ? "wss:" : "ws:";
  // ws:// vs wss:// mismatch is a common deployment bug (see
  // docs/03_DEPLOYMENT_GUIDE.txt Part 1 Step 5) -- deriving the scheme
  // from the configured API_URL's own protocol, rather than hardcoding
  // one, is what keeps this correct automatically after deployment.
  return `${scheme}//${httpUrl.host}${path}`;
}

export { API_URL };
