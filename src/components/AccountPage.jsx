import React, { useState, useEffect } from "react";
import { doc, getDoc, setDoc } from "firebase/firestore";
import { db } from "../firebase";
import "./AccountPage.css";

function AccountPage({ user }) {
  const [limit, setLimit] = useState("");
  const [saving, setSaving] = useState(false);
  const [savedMsg, setSavedMsg] = useState("");
  const [profile, setProfile] = useState(null);

  // Always keep Firestore doc's email/displayName in sync for backend notifications
  useEffect(() => {
    if (user?.uid && user?.email) {
      setDoc(doc(db, "users", user.uid), {
        email: user.email,
        displayName: user.displayName || "",
      }, { merge: true });
    }
  }, [user]);

  // Load current screen time limit (and possible other profile fields)
  useEffect(() => {
    if (user?.uid) {
      getDoc(doc(db, "users", user.uid)).then(snap => {
        if (snap.exists()) {
          const data = snap.data();
          if ("screenTimeLimit" in data) setLimit(data.screenTimeLimit);
          setProfile(data);
        }
      });
    }
  }, [user]);

  // Save new limit to Firestore (also always resyncs profile info)
  const handleLimitSave = async (e) => {
    e.preventDefault();
    setSaving(true);
    await setDoc(
      doc(db, "users", user.uid),
      {
        screenTimeLimit: limit,
        email: user.email,
        displayName: user.displayName || "",
      },
      { merge: true }
    );
    setSavedMsg("Limit saved!");
    setSaving(false);
    setTimeout(() => setSavedMsg(""), 1800);
  };

  if (!user) return <div className="account-box"><p>No user logged in.</p></div>;

  const displayName = user.displayName || user.email || "User";
  const email = user.email || "N/A";
  const uid = user.uid;
  const photoURL =
    user.photoURL && user.photoURL.trim() !== ""
      ? user.photoURL
      : `https://ui-avatars.com/api/?name=${encodeURIComponent(displayName)}&background=2067d8&color=fff`;
  const creationTime = user.metadata?.creationTime || "N/A";

  return (
    <div className="account-box">
      <div className="account-profile-img-wrapper">
        <img
          className="account-profile-img"
          src={photoURL}
          alt="Profile"
          onError={e => {
            e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(displayName)}&background=2067d8&color=fff`;
          }}
        />
      </div>
      <h2>Account Details</h2>
      <p><strong>Name:</strong> {displayName}</p>
      <p><strong>Email:</strong> {email}</p>
      <p><strong>User ID:</strong> {uid}</p>
      <p><strong>Sign Up Date:</strong> {creationTime}</p>

      <form onSubmit={handleLimitSave} className="screen-limit-form">
        <label htmlFor="limit" style={{ marginTop: "1.3rem" }}>
          <strong>Screen Time Limit (per day)</strong> (minutes):
        </label>
        <input
          type="number"
          id="limit"
          min="1"
          step="1"
          value={limit}
          onChange={e => setLimit(e.target.value)}
          style={{ maxWidth: 120, marginTop: 6, marginBottom: 8 }}
          placeholder="e.g. 120"
        />
        <button
          className="btn-cta"
          disabled={saving || !limit || isNaN(Number(limit))}
          style={{ marginLeft: 8 }}
        >
          {saving ? "Saving..." : "Save"}
        </button>
        {savedMsg && <span style={{ color: "#24a271", marginLeft: 10 }}>{savedMsg}</span>}
      </form>
      {limit && (
        <div style={{ marginTop: 6, fontSize: "1rem", color: "#174ba0" }}>
          Limit set to <b>{limit} min/day</b>
        </div>
      )}
    </div>
  );
}

export default AccountPage;
