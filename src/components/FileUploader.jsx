import React from "react";
import Papa from "papaparse";
import "./FileUploader.css";

function FileUploader({ setData, setColumns, setFileInfo }) {
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: function (results) {
        const rows = results.data;
        const headers = results.meta.fields;

        setData(rows);
        setColumns(headers);

        setFileInfo({
          fileName: file.name,
          rowCount: rows.length,
          columnCount: headers.length,
          headers: headers,
        });

        alert(`✅ File uploaded!\n${rows.length} rows with ${headers.length} columns`);
      }
    });
  };

  return (
    <div className="fileuploader-box">
      <h2>Upload CSV Dataset</h2>
      <label htmlFor="upload" className="fileuploader-label">
        <span className="fileuploader-custom">Choose CSV File</span>
        <input
          id="upload"
          className="fileuploader-input"
          type="file"
          accept=".csv"
          onChange={handleFileUpload}
        />
      </label>
      <div className="fileuploader-note">
        Upload CSV files with proper headers to unlock full chart and table features.
      </div>
    </div>
  );
}

export default FileUploader;
