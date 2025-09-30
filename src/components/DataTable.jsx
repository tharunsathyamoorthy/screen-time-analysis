import React from "react";
import "./DataTable.css";

function DataTable({ data = [], columns = [] }) {
  if (!data.length || !columns.length) {
    return (
      <div className="data-table-section">
        <h2>Raw Data</h2>
        <p>No data available. Please upload a CSV file to view raw data.</p>
      </div>
    );
  }

  return (
    <div className="data-table-section">
      <h2>Raw Data</h2>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>#</th> {/* Optional row index */}
              {columns.map((col) => (
                <th key={col}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, index) => (
              <tr key={index}>
                <td>{index + 1}</td>
                {columns.map((col) => (
                  <td key={col}>{row[col]}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default DataTable;
