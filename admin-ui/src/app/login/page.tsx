'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Stethoscope, LogIn } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const { login } = useAuth();
  const router = useRouter();
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoggingIn(true);
    const success = await login(email, password);
    if (success) {
      router.push('/dashboard');
      toast({
        title: "Login Successful",
        description: "Welcome back!",
      });
    } else {
      toast({
        title: "Login Failed",
        description: "Invalid email or password.",
        variant: "destructive",
      });
      setIsLoggingIn(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md shadow-xl">
        <CardHeader className="space-y-1 text-center">
          <div className="flex justify-center items-center gap-2 mb-4">
            <Stethoscope className="h-10 w-10 text-primary" />
            <CardTitle className="text-3xl font-bold tracking-tight">MediAdmin</CardTitle>
          </div>
          <CardDescription>Enter your credentials to access the admin panel</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="email">Email & Username</Label>
              <Input
                id="email"
                type="text"
                placeholder="admin@medadmin.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="bg-input border-border focus:ring-primary"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                placeholder="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="bg-input border-border focus:ring-primary"
              />
            </div>
            <Button type="submit" className="w-full" disabled={isLoggingIn}>
              {isLoggingIn ? (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary-foreground mr-2"></div>
              ) : (
                <LogIn className="mr-2 h-4 w-4" />
              )}
              {isLoggingIn ? 'Logging In...' : 'Login'}
            </Button>
          </form>
        </CardContent>
        <CardFooter className="flex justify-center text-xs text-muted-foreground">
           <p>&copy; {new Date().getFullYear()} MediAdmin. All rights reserved.</p>
        </CardFooter>
      </Card>
    </div>
  );
}
