import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import "./App.css";
import { ChatPage } from "./pages/ChatPage";
import { StudioPage } from "./pages/StudioPage";

function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <header className="top-nav">
          <span className="brand">Advanced RAG</span>
          <nav>
            <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
              Chat
            </NavLink>
            <NavLink to="/studio" className={({ isActive }) => (isActive ? "active" : "")}>
              Studio
            </NavLink>
          </nav>
        </header>
        <main className="main-area">
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/studio" element={<StudioPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
