
// @ts-nocheck
"use client";

import React, { useEffect, useState, Suspense } from 'react';
import type { DashboardMetrics, DataAnalysisDataPoint } from "@/lib/mockData"; // Re-using existing types for now
import { MetricsCard } from "@/components/dashboard/MetricsCard";
import { DataAnalysisChart } from "@/components/dashboard/DataAnalysisChart";
import { BriefcaseMedical, Users, Video, BarChart3, AlertCircle } from "lucide-react";
import AppLayout from "@/components/layout/AppLayout";
import { useAuth } from '@/context/AuthContext';
import { useToast } from '@/hooks/use-toast';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';

import { API_BASE_URL } from '@/config/api';

function DashboardContent() {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [chartData, setChartData] = useState<DataAnalysisDataPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { currentUser } = useAuth();
  const { toast } = useToast();

  const fetchData = async () => {
    if (!currentUser?.id) {
      setError("User not authenticated.");
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const [metricsResponse, chartResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/management/dashboard/metrics?admin_doctor_id=${currentUser.id}`),
        fetch(`${API_BASE_URL}/management/dashboard/analysis-trends?admin_doctor_id=${currentUser.id}`)
      ]);

      if (!metricsResponse.ok) {
        const errData = await metricsResponse.json();
        throw new Error(`Failed to fetch metrics: ${metricsResponse.status} ${errData.detail || metricsResponse.statusText}`);
      }
      const metricsData: DashboardMetrics = await metricsResponse.json();
      setMetrics(metricsData);

      if (!chartResponse.ok) {
        const errData = await chartResponse.json();
        throw new Error(`Failed to fetch chart data: ${chartResponse.status} ${errData.detail || chartResponse.statusText}`);
      }
      const chartDataFromServer: DataAnalysisDataPoint[] = await chartResponse.json();
      setChartData(chartDataFromServer);

    } catch (e) {
      console.error("Dashboard fetch error:", e);
      setError((e as Error).message);
      toast({
        title: "Error Fetching Dashboard Data",
        description: (e as Error).message,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [currentUser?.id]);


  if (isLoading) {
    return (
      <div className="flex flex-col gap-6">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">Dashboard</h1>
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
        <ChartSkeleton />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-destructive p-6 rounded-lg border border-destructive/50 bg-destructive/10">
        <AlertCircle className="w-16 h-16" />
        <h2 className="text-2xl font-semibold">Error Loading Dashboard</h2>
        <p className="text-center max-w-md">{error}</p>
        <Button onClick={fetchData} variant="destructive">Retry</Button>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
        <p>No dashboard data available.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-3xl font-bold tracking-tight text-foreground">Dashboard</h1>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricsCard
          title="Total Doctors"
          value={metrics.doctorCount}
          icon={BriefcaseMedical}
          description="Number of active doctors"
        />
        <MetricsCard
          title="Total Patients"
          value={metrics.patientCount}
          icon={Users}
          description="Registered patients in system"
        />
        <MetricsCard
          title="Uploaded Videos"
          value={metrics.videoCount}
          icon={Video}
          description="Consultation videos"
        />
        <MetricsCard
          title="Data Analyses"
          value={metrics.dataAnalysisCount}
          icon={BarChart3}
          description="Analyses performed"
        />
      </div>
      {chartData.length > 0 ? (
        <DataAnalysisChart data={chartData} />
      ) : (
        <div className="p-4 text-center text-muted-foreground bg-card rounded-lg shadow-md h-[350px] flex items-center justify-center">
          No analysis trend data available.
        </div>
      )}
    </div>
  );
}

const CardSkeleton = () => (
  <div className="p-4 bg-card rounded-lg shadow-md space-y-2">
    <Skeleton className="h-5 w-3/4" />
    <Skeleton className="h-8 w-1/2" />
    <Skeleton className="h-4 w-full" />
  </div>
);

const ChartSkeleton = () => (
  <div className="p-4 bg-card rounded-lg shadow-md">
    <Skeleton className="h-5 w-1/3 mb-2" />
    <Skeleton className="h-4 w-1/2 mb-4" />
    <Skeleton className="h-[300px] w-full" />
  </div>
);


export default function DashboardPage() {
  return (
    <AppLayout>
      <Suspense fallback={<div className="flex justify-center items-center h-64 text-muted-foreground">Loading dashboard...</div>}>
        <DashboardContent />
      </Suspense>
    </AppLayout>
  );
}
