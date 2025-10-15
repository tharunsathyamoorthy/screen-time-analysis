import React from "react";
import "./Dashboard.css";

function Dashboard({ data = [], fileInfo, columns = [] }) {
  // Force use of 'Daily_Screen_Time_Hours' for average calculation
  const usageCol = "Daily_Screen_Time_Hours";
  
  // Accurate average (Pandas-style): mean of valid finite entries in Daily_Screen_Time_Hours
  const validRows = data.filter(
    row => usageCol && row[usageCol] !== null && row[usageCol] !== "" && isFinite(Number(row[usageCol]))
  );
  const avgTime = usageCol && validRows.length > 0
    ? (
        validRows.reduce((sum, row) => sum + Number(row[usageCol]), 0) / validRows.length
      ).toFixed(2)
    : "N/A";

  return (
    <div className="dashboard-section">
      <h2>Summary</h2>
      <ul>
        <li>Average Screen Time: <b>{avgTime}</b> hrs/day</li>
        <li>Total Participants: <b>{data.length}</b></li>
      </ul>
      {fileInfo ? (
        <div className="file-summary">
          <h3>📄 File Info</h3>
          <p><strong>File Name:</strong> {fileInfo.fileName}</p>
          <p><strong>Rows:</strong> {fileInfo.rowCount}</p>
          <p><strong>Columns:</strong> {fileInfo.columnCount}</p>
          <p><strong>Headers:</strong> {fileInfo.headers.join(", ")}</p>
        </div>
      ) : (
        <p>No file uploaded yet.</p>
      )}
    </div>
  );
}

export default Dashboard;
