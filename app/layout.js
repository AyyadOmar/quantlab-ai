import "./globals.css";

export const metadata = {
  title: "QuantLab AI",
  description: "Quantitative machine learning research platform for walk-forward validation and benchmark-aware backtesting.",
  icons: {
    icon: "/favicon.png",
    shortcut: "/favicon.png",
    apple: "/favicon.png"
  }
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
