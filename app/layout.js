import "./globals.css";

export const metadata = {
  title: "QuantLab AI",
  description: "Quantitative machine learning research platform for walk-forward validation and benchmark-aware backtesting."
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
