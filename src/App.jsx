import React, { useEffect, useState } from "react";
import { onAuthStateChanged, signOut } from "firebase/auth";
import { auth } from "./firebase";

import SignUpPage from "./components/SignUpPage";
import SignInPage from "./components/SignInPage";
import Sidebar from "./components/Sidebar";
import Dashboard from "./components/Dashboard";
import AccountPage from "./components/AccountPage";
import FileUploader from "./components/FileUploader";
import Charts from "./components/Charts";
import DataTable from "./components/DataTable";
import AnalysisPage from "./components/AnalysisPage"; // Correct import reflecting actual filename

import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "./App.css";

function App() {
  const [user, setUser] = useState(null);
  const [section, setSection] = useState("dashboard");
  const [data, setData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [fileInfo, setFileInfo] = useState(null);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setSection(currentUser ? "dashboard" : "signin");
    });
    return () => unsubscribe();
  }, []);

  const handleSignOut = async () => {
    try {
      await signOut(auth);
      setUser(null);
      setData([]);
      setColumns([]);
      setFileInfo(null);
    } catch (err) {
      console.error("Sign out error:", err);
    }
  };

  if (!user) {
    if (section === "signup") {
      return <SignUpPage onSignUp={() => setSection("account")} />;
    }
    return (
      <>
        <SignInPage
          onSignIn={() => setSection("account")}
          onGoToSignUp={() => setSection("signup")}
        />
        <ToastContainer position="bottom-center" theme="light" autoClose={3000} />
      </>
    );
  }

  return (
    <div className="main-layout">
      <Sidebar section={section} setSection={setSection} onSignOut={handleSignOut} />
      <div className="content">
        {section === "dashboard" && (
          <Dashboard data={data} columns={columns} fileInfo={fileInfo} />
        )}
        {section === "upload" && (
          <FileUploader setData={setData} setColumns={setColumns} setFileInfo={setFileInfo} />
        )}
        {section === "charts" && <Charts data={data} columns={columns} />}
        {section === "data" && <DataTable data={data} columns={columns} />}
        {section === "ml" && <AnalysisPage />}  {/* Use AnalysisPage component */}
        {section === "account" && <AccountPage user={user} />}
      </div>
      <ToastContainer position="bottom-center" theme="light" autoClose={3000} />
    </div>
  );
}

export default App;
