import React, { useState, useEffect } from "react";
import "./AnalysisPage.css";

function AnalysisPage() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("");
  const [age, setAge] = useState("");
  const [ageAnalysis, setAgeAnalysis] = useState(null);
  const [ageLoading, setAgeLoading] = useState(false);

  useEffect(() => {
    const savedResult = localStorage.getItem("analysisResult");
    const savedFileName = localStorage.getItem("uploadedFileName");
    if (savedResult) setResult(JSON.parse(savedResult));
    if (savedFileName) setFileName(savedFileName);
  }, []);

  const handleFileChange = (e) => {
    const newFile = e.target.files[0];
    setFile(newFile);
    setResult(null);
    setError("");
    if (newFile) {
      setFileName(newFile.name);
      localStorage.setItem("uploadedFileName", newFile.name);
      localStorage.removeItem("analysisResult");
    }
  };

  const handleRunAnalysis = async () => {
    if (!file) {
      setError("Please upload a CSV file.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        let errMsg = "Server error. Please try again.";
        try {
          const errData = await res.json();
          errMsg = errData.error || errMsg;
        } catch {}
        throw new Error(errMsg);
      }
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
      localStorage.setItem("analysisResult", JSON.stringify(data));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAgeAnalysis = async () => {
    if (!age) {
      setError("Please enter an age for analysis.");
      return;
    }
    setAgeLoading(true);
    setError("");
    setAgeAnalysis(null);

    try {
      const res = await fetch("http://localhost:5000/analyze_age", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ age: Number(age) }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setAgeAnalysis(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setAgeLoading(false);
    }
  };

  const safeToFixed = (num, digits = 2, fallback = "N/A") =>
    typeof num === "number" && !isNaN(num) ? num.toFixed(digits) : fallback;

  return (
    <div className="center-outer">
      <div className="analysis-section">
        <h2>Screen Time Vision Risk Analysis</h2>
        <p>
          Upload your <b>screen time survey</b> CSV file to analyze vision risk.
        </p>

        {/* Upload Section */}
        <div className="upload-area">
          <input type="file" accept=".csv" onChange={handleFileChange} />
          {fileName && (
            <div>
              Selected File: <b>{fileName}</b>
            </div>
          )}
          <button
            className="primary-btn"
            disabled={loading}
            onClick={handleRunAnalysis}
          >
            {loading ? "Analyzing..." : "Run Analysis"}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}

        {/* Display Results */}
        {result && !result.error && (
          <div className="results">
            <h3>Summary:</h3>
            <p>
              <b>Screen Time Column:</b> {result.screen_time_column ?? "N/A"}
            </p>
            <p>
              <b>Average Screen Time:</b>{" "}
              {safeToFixed(result?.average_screen_time_hours)} hours/day
            </p>
            <p>
              <b>
                % Exceeding Recommendation (
                {result?.recommendation_hours !== undefined
                  ? safeToFixed(result?.recommendation_hours)
                  : "N/A"}{" "}
                hrs):
              </b>{" "}
              {safeToFixed(result?.percentage_exceeding_recommendation)}%
            </p>

            <h3>Vision Risk Distribution:</h3>
            <ul>
              {[
                "Low Exposure - Healthy Visual Ergonomics",
                "Moderate Exposure - Normal Ocular Endurance",
                "High Exposure - Digital Eye Strain Risk",
              ].map((category) => (
                <li key={category}>
                  <b>{category}:</b>{" "}
                  {result?.health_impact_trends &&
                  result?.health_impact_trends[category] !== undefined
                    ? result.health_impact_trends[category]
                    : 0}
                </li>
              ))}
            </ul>

            {/* Charts */}
            {result?.charts?.visualizations && (
              <>
                <h3>Visualizations:</h3>
                {Array.isArray(result.charts.visualizations)
                  ? result.charts.visualizations.map((imgUrl, idx) => (
                      <img
                        key={idx}
                        src={imgUrl}
                        alt={`Visualization ${idx + 1}`}
                        style={{
                          maxWidth: "100%",
                          marginTop: 20,
                          borderRadius: 10,
                          boxShadow: "0 1px 8px rgba(0,0,0,0.1)",
                        }}
                      />
                    ))
                  : (
                    <img
                      src={result.charts.visualizations}
                      alt="Data Visualizations"
                      style={{
                        maxWidth: "100%",
                        marginTop: 20,
                        borderRadius: 10,
                        boxShadow: "0 1px 8px rgba(0,0,0,0.1)",
                      }}
                    />
                  )}
              </>
            )}

            {/* Precautions */}
            {result?.vision_precautions && (
              <div style={{ marginTop: "30px" }}>
                <h3>Precautions for Each Vision Risk Category:</h3>
                {Object.entries(result.vision_precautions).map(
                  ([category, points]) => (
                    <div key={category} style={{ marginBottom: "20px" }}>
                      <b>{category}</b>
                      <ul>
                        {points.map((pt, idx) => (
                          <li key={idx}>{pt}</li>
                        ))}
                      </ul>
                    </div>
                  )
                )}
              </div>
            )}

            {/* Device Summary */}
            {result?.device_usage_summary && (
              <>
                <h3>Device/App Usage Summary (if available):</h3>
                <pre className="device-summary">
                  {JSON.stringify(result.device_usage_summary, null, 2)}
                </pre>
              </>
            )}

            {/* Processed Data Table */}
            <h3>Sample Processed Data (First 10 rows):</h3>
            <div style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    {result?.processed_data_sample &&
                      result.processed_data_sample.length > 0 &&
                      Object.keys(result.processed_data_sample[0]).map((col) => (
                        <th key={col}>{col}</th>
                      ))}
                  </tr>
                </thead>
                <tbody>
                  {result?.processed_data_sample &&
                    result.processed_data_sample.map((row, idx) => (
                      <tr key={idx}>
                        {Object.values(row).map((val, i) => (
                          <td key={i}>
                            {val === null || val === undefined
                              ? "-"
                              : val.toString()}
                          </td>
                        ))}
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>

            {/* 🔹 Age Group Analysis Section */}
            <div className="age-section">
              <h3>Analyze Vision Status by Age Group</h3>
              <div style={{ marginTop: "10px" }}>
                <input
                  type="number"
                  placeholder="Enter Age"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  style={{ padding: "8px", marginRight: "10px" }}
                />
                <button
                  className="primary-btn"
                  disabled={ageLoading}
                  onClick={handleAgeAnalysis}
                >
                  {ageLoading ? "Analyzing..." : "Analyze by Age"}
                </button>
              </div>

              {ageAnalysis && (
                <div style={{ marginTop: "25px" }}>
                  <h4>Age Group: {ageAnalysis.age_group}</h4>

                  {ageAnalysis.records && ageAnalysis.records.length > 0 ? (
                    <>
                      <h4>People in this Age Group:</h4>
                      <table>
                        <thead>
                          <tr>
                            {Object.keys(ageAnalysis.records[0]).map((col) => (
                              <th key={col}>{col}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {ageAnalysis.records.map((row, idx) => (
                            <tr key={idx}>
                              {Object.values(row).map((val, i) => (
                                <td key={i}>{val}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>

                      {ageAnalysis.chart_url && (
                        <img
                          src={ageAnalysis.chart_url}
                          alt="Age Group Vision Distribution"
                          style={{
                            maxWidth: "100%",
                            marginTop: "20px",
                            borderRadius: "10px",
                            boxShadow: "0 1px 8px rgba(0,0,0,0.1)",
                          }}
                        />
                      )}
                    </>
                  ) : (
                    <p style={{ color: "gray", marginTop: "15px" }}>
                      No records found for this age group.
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default AnalysisPage;
