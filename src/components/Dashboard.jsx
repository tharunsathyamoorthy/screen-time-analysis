import React from "react";
import "./Dashboard.css";

function Dashboard({ data = [], fileInfo, columns = [] }) {
  // ✅ Smarter usage column detection based on your real column name
  const usageCol = columns.find(col =>
    ["screen time", "screentime", "avg_daily_screen_time_hr", "screen", "usage", "minutes"].some(keyword =>
      col.toLowerCase().replace(/_/g, " ").includes(keyword)
    )
  );

  // ✅ Average time calculation
  const avgTime = usageCol
    ? (
        data.reduce((sum, row) => sum + (Number(row[usageCol]) || 0), 0) / data.length
      ).toFixed(2)
    : "N/A";

  return (
    <div className="dashboard-section">
      <h2>Summary</h2>

      <ul>
        <li>Average Screen Time: <b>{avgTime}</b> hrs/day</li>
        <li>Total Participants: <b>{data.length}</b></li>
      </ul>

      {/* Show file details if available */}
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
