// pages/LoginPage.jsx
// --------------------
// Two modes: sign in (existing account, JWT via /auth/login) and create
// account (POST /users, which -- per Phase C's design -- returns a
// watcher_token). That token is shown ONCE, right here, with an
// explicit instruction to paste it into the local watcher's config: the
// dashboard itself never stores or reuses it (see api/client.js's
// docstring) -- it exists only to hand off to the watcher, matching the
// two-part deployment architecture (Section 12).

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, setToken } from "../api/client";

export default function LoginPage() {
  const navigate = useNavigate();
  const [mode, setMode] = useState("signin"); // "signin" | "signup"
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [watcherToken, setWatcherToken] = useState(null);
  const [busy, setBusy] = useState(false);

  async function signIn(targetEmail, targetPassword) {
    const { access_token } = await api.login(targetEmail, targetPassword);
    setToken(access_token);
    navigate("/planner");
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      if (mode === "signup") {
        const user = await api.signUp(name, email, password);
        setWatcherToken(user.watcher_token);
      } else {
        await signIn(email, password);
      }
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  if (watcherToken) {
    return (
      <div className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-6">
        <h1 className="font-display text-2xl text-ink-50">Account created</h1>
        <p className="mt-3 text-ink-200">
          Paste this token into your local watcher's <code className="text-ink-50">.env</code> file (or however
          your watcher is configured) so it can report to this dashboard. This is shown once -- copy it now.
        </p>
        <div className="mt-4 rounded-lg border border-ink-600 bg-ink-900 p-4 font-mono text-sm text-signal-amber break-all">
          {watcherToken}
        </div>
        <button
          onClick={() => signIn(email, password)}
          className="mt-6 rounded-md bg-signal-amber px-4 py-2 font-medium text-ink-950 transition-opacity hover:opacity-90"
        >
          Continue to dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-6">
      <h1 className="font-display text-3xl text-ink-50">AI Study Invigilator</h1>
      <p className="mt-2 text-ink-400">{mode === "signup" ? "Create your account" : "Sign in to your dashboard"}</p>

      <form onSubmit={handleSubmit} className="mt-8 space-y-4">
        {mode === "signup" && (
          <Field label="Name">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              className="w-full rounded-md border border-ink-600 bg-ink-900 px-3 py-2 text-ink-50 outline-none focus:border-signal-amber"
            />
          </Field>
        )}
        <Field label="Email">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="w-full rounded-md border border-ink-600 bg-ink-900 px-3 py-2 text-ink-50 outline-none focus:border-signal-amber"
          />
        </Field>
        <Field label="Password">
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            className="w-full rounded-md border border-ink-600 bg-ink-900 px-3 py-2 text-ink-50 outline-none focus:border-signal-amber"
          />
        </Field>

        {error && <p className="text-sm text-verdict-shorter">{error}</p>}

        <button
          type="submit"
          disabled={busy}
          className="w-full rounded-md bg-signal-amber px-4 py-2 font-medium text-ink-950 transition-opacity hover:opacity-90 disabled:opacity-50"
        >
          {busy ? "Working…" : mode === "signup" ? "Create account" : "Sign in"}
        </button>
      </form>

      <button
        onClick={() => {
          setMode(mode === "signup" ? "signin" : "signup");
          setError("");
        }}
        className="mt-6 text-sm text-ink-400 hover:text-ink-100"
      >
        {mode === "signup" ? "Already have an account? Sign in" : "New here? Create an account"}
      </button>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="block">
      <span className="mb-1 block text-sm text-ink-200">{label}</span>
      {children}
    </label>
  );
}
