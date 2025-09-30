import React, { useState, useEffect } from "react";
import { Bar } from "react-chartjs-2";
import "./Charts.css";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function Charts({ data = [], columns = [] }) {
  const [selected, setSelected] = useState("");

  useEffect(() => {
    if (columns.length > 0 && !selected) {
      setSelected(columns[0]);
    }
  }, [columns, selected]);

  if (!data.length || !selected) {
    return <p style={{ margin: 40, color: "#888", fontWeight: 500 }}>Upload a CSV file to see charts.</p>;
  }

  // Numeric/categorical detection
  const values = data.map(row => row[selected]).filter(val => !!val);
  const allNumeric = values.every(val => !isNaN(Number(val)));

  const counts = {};
  values.forEach(val => {
    const key = allNumeric ? Number(val) : val;
    counts[key] = (counts[key] || 0) + 1;
  });

  const chartData = {
    labels: Object.keys(counts),
    datasets: [{
      label: selected,
      data: Object.values(counts),
      backgroundColor: allNumeric ? "#2067d8" : "#6195ed"
    }]
  };

  // Axis label determination
  const xAxisLabel = selected;
  const yAxisLabel = allNumeric ? "Count" : "Frequency";

  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: {
        display: false,
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: xAxisLabel,
          font: { size: 15, weight: "bold" }
        }
      },
      y: {
        title: {
          display: true,
          text: yAxisLabel,
          font: { size: 15, weight: "bold" }
        },
        beginAtZero: true
      }
    }
  };

  return (
    <div className="chart-section">
      <h2>Column Chart</h2>
      <select
        value={selected}
        onChange={e => setSelected(e.target.value)}
        style={{ marginBottom: 16 }}
      >
        {columns.map(col => (
          <option key={col} value={col}>{col}</option>
        ))}
      </select>
      <div className="chart-container" style={{ minHeight: 320 }}>
        <Bar data={chartData} options={options} />
      </div>
    </div>
  );
}

export default Charts;
