import React from "react";
import "./Sidebar.css";

function Sidebar({ section, setSection, onSignOut }) {
  return (
    <nav className="sidebar">
      <button className={section === "dashboard" ? "active" : ""} onClick={() => setSection("dashboard")}>Dashboard</button>
      <button className={section === "upload" ? "active" : ""} onClick={() => setSection("upload")}>Upload Data</button>
      <button className={section === "charts" ? "active" : ""} onClick={() => setSection("charts")}>Charts</button>
      <button className={section === "data" ? "active" : ""} onClick={() => setSection("data")}>Raw Data</button>
      <button className={section === "ml" ? "active" : ""} onClick={() => setSection("ml")}>Analysis</button>
      <button className={section === "account" ? "active" : ""} onClick={() => setSection("account")}>Account</button>
      <button className="signout-btn" onClick={onSignOut}>Sign Out</button>
    </nav>
  );
}

export default Sidebar;
