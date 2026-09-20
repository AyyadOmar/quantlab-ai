import "./globals.css";

// Vercel provides these trusted deployment hosts. Local previews use localhost.
const deploymentHost = process.env.VERCEL_PROJECT_PRODUCTION_URL || process.env.VERCEL_URL;
const description = "Stock prediction measured against always-up. Explore transparent model comparisons, chronological evaluation, and the complete QuantLab AI research record.";

export const metadata = {
  metadataBase: new URL(deploymentHost ? `https://${deploymentHost}` : "http://localhost:3000"),
  title: "QuantLab AI — Stock prediction, measured",
  description,
  openGraph: {
    title: "QuantLab AI — Stock prediction, measured",
    description,
    siteName: "QuantLab AI",
    type: "website",
    locale: "en_US",
    images: [{ url: "/og.png", width: 1730, height: 909, alt: "QuantLab AI: Stock prediction. Measured against the baseline." }]
  },
  twitter: {
    card: "summary_large_image",
    title: "QuantLab AI — Stock prediction, measured",
    description,
    images: ["/og.png"]
  },
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
