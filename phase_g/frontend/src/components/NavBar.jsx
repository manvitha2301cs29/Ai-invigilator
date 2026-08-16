// components/NavBar.jsx
import { NavLink, useNavigate } from "react-router-dom";
import { setToken } from "../api/client";

const LINKS = [
  { to: "/planner", label: "Planner" },
  { to: "/live", label: "Live status" },
  { to: "/history", label: "History" },
];

export default function NavBar() {
  const navigate = useNavigate();

  function signOut() {
    setToken(null);
    navigate("/login");
  }

  return (
    <header className="border-b border-ink-700 bg-ink-900/60 backdrop-blur">
      <div className="mx-auto flex max-w-4xl items-center justify-between px-6 py-4">
        <div className="font-display text-lg tracking-tight text-ink-50">AI Study Invigilator</div>
        <nav className="flex items-center gap-1">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) =>
                `rounded-md px-3 py-1.5 text-sm transition-colors ${
                  isActive ? "bg-ink-700 text-ink-50" : "text-ink-200 hover:bg-ink-800 hover:text-ink-50"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
          <button
            onClick={signOut}
            className="ml-2 rounded-md px-3 py-1.5 text-sm text-ink-400 transition-colors hover:bg-ink-800 hover:text-ink-100"
          >
            Sign out
          </button>
        </nav>
      </div>
    </header>
  );
}
