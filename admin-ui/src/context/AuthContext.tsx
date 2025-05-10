'use client';

import type { User } from '@/types';
import { useRouter } from 'next/navigation';
import type { ReactNode} from 'react';
import { createContext, useContext, useEffect, useState } from 'react';
import { API_BASE_URL } from '@/config/api';

interface AuthContextType {
  currentUser: User | null;
  isLoading: boolean;
  login: (email: string, pass: string) => Promise<boolean>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    // Check for stored user on initial load
    try {
      const storedUser = localStorage.getItem('currentUser');
      if (storedUser) {
        setCurrentUser(JSON.parse(storedUser));
      }
    } catch (error) {
      // console.error("Failed to parse user from localStorage", error);
      localStorage.removeItem('currentUser');
    }
    setIsLoading(false);
  }, []);

  const login = async (email: string, pass: string): Promise<boolean> => {
    setIsLoading(true);
    // This is a mock authentication.
    // In a real application, you would call your backend's login endpoint.
    // For example: POST /api/v1/doctors/login or /api/v1/management/login
    // if (email === 'admin@medadmin.com' && pass === 'password') {
    //   const user: User = {
    //     id: '1', // Assuming admin user has ID '1' as often is the case in DBs
    //     name: 'Admin User', // Corresponds to username in backend
    //     email: 'admin@medadmin.com',
    //     avatarUrl: 'https://picsum.photos/seed/admin/40/40',
    //     role_id: 1, // Admin role
    //   };
    //   setCurrentUser(user);
    //   localStorage.setItem('currentUser', JSON.stringify(user));
    //   setIsLoading(false);
    //   router.push('/dashboard'); // Redirect after successful login
    //   return true;
    // }
    // Simulate API call for non-admin/failed login
    try {
      const response = await fetch(`${API_BASE_URL}/management/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email, password: pass }),
      });
      if (response.ok) {
        const data = await response.json();
        if (data.doctor) {
          const loggedInUser: User = {
            id: data.doctor.id.toString(),
            name: data.doctor.username,
            email: data.doctor.email,
                role_id: data.doctor.role_id,
            // avatarUrl might not be available from this endpoint
          };
          setCurrentUser(loggedInUser);
          localStorage.setItem('currentUser', JSON.stringify(loggedInUser));
          setIsLoading(false);
          router.push('/dashboard');
          return true;
        }
      }
    } catch (error) {
      console.error("Login API call failed:", error);
    }
    setIsLoading(false);
    return false;
  };

  const logout = () => {
    setCurrentUser(null);
    localStorage.removeItem('currentUser');
    router.push('/login');
  };

  return (
    <AuthContext.Provider value={{ currentUser, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
