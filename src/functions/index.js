const functions = require("firebase-functions");
const admin = require("firebase-admin");
const nodemailer = require("nodemailer");

admin.initializeApp();

// Use an App Password for Gmail, or your SMTP service.
const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: "yourgmail@gmail.com",
    pass: "your-app-password"
  }
});

exports.sendLimitEmail = functions.firestore
  .document('users/{userId}')
  .onWrite(async (change, context) => {
    const before = change.before.data() || {};
    const after = change.after.data() || {};

    if (
      after.currentScreenTime > after.screenTimeLimit &&
      (before.currentScreenTime || 0) <= (before.screenTimeLimit || 0) &&
      after.email
    ) {
      await transporter.sendMail({
        to: after.email,
        subject: "Screen Time Limit Exceeded!",
        text: `Hi ${after.displayName || "User"},\n\nYou have crossed your daily screen time limit of ${after.screenTimeLimit} minutes today.`,
      });
    }
  });
