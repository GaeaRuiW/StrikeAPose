import type { Metadata } from 'next';
import { Inter as Geist, Inter as Geist_Mono } from 'next/font/google'; // Using Inter as a placeholder for Geist
import './globals.css';
import { Toaster } from "@/components/ui/toaster";
import AppLayout from "@/components/layout/AppLayout";
import { AuthProvider } from '@/context/AuthContext';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'MediAdmin',
  description: 'Medical Administration Panel',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <AuthProvider>
          {/* AppLayout is conditionally rendered based on route in specific page layouts or here */}
          {/* For now, children might include AppLayout or LoginPage directly */}
          {children}
          <Toaster />
        </AuthProvider>
      </body>
    </html>
  );
}
