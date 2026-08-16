import { Navigate, Route, Routes } from "react-router-dom";
import { getToken } from "./api/client";
import NavBar from "./components/NavBar";
import LoginPage from "./pages/LoginPage";
import PlannerPage from "./pages/PlannerPage";
import LiveStatusPage from "./pages/LiveStatusPage";
import SummaryPage from "./pages/SummaryPage";
import HistoryPage from "./pages/HistoryPage";

function RequireAuth({ children }) {
  if (!getToken()) return <Navigate to="/login" replace />;
  return children;
}

function DashboardLayout({ children }) {
  return (
    <div className="min-h-screen">
      <NavBar />
      <main>{children}</main>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/planner"
        element={
          <RequireAuth>
            <DashboardLayout>
              <PlannerPage />
            </DashboardLayout>
          </RequireAuth>
        }
      />
      <Route
        path="/live"
        element={
          <RequireAuth>
            <DashboardLayout>
              <LiveStatusPage />
            </DashboardLayout>
          </RequireAuth>
        }
      />
      <Route
        path="/summary/:blockId?"
        element={
          <RequireAuth>
            <DashboardLayout>
              <SummaryPage />
            </DashboardLayout>
          </RequireAuth>
        }
      />
      <Route
        path="/history"
        element={
          <RequireAuth>
            <DashboardLayout>
              <HistoryPage />
            </DashboardLayout>
          </RequireAuth>
        }
      />
      <Route path="*" element={<Navigate to="/planner" replace />} />
    </Routes>
  );
}
