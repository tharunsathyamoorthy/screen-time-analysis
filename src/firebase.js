import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";
import { getFirestore } from "firebase/firestore"; // ✅ Add this

const firebaseConfig = {
  apiKey: "AIzaSyBBGBP-hp5kxiBMDiFOy8k3ZInIkoVBdxs",
  authDomain: "screentime-964f1.firebaseapp.com",
  projectId: "screentime-964f1",
  storageBucket: "screentime-964f1.appspot.com",
  messagingSenderId: "697389234858",
  appId: "1:697389234858:web:1074abba5f88072cb76ed3",
  measurementId: "G-RS14F9H42W"
};

const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const analytics = getAnalytics(app);
export const db = getFirestore(app); // ✅ Add this line to fix the error
