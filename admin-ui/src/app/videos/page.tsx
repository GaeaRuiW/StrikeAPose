// @ts-nocheck
"use client";

import React, { useEffect, useMemo, useState, Suspense, useCallback } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import type { Video, Patient } from '@/types';
import { VideosTable } from '@/components/videos/VideosTable';
import { VideoPlayerModal } from '@/components/videos/VideoPlayerModal'; // Import the modal
import { Input } from '@/components/ui/input';
import { Search, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { useToast } from "@/hooks/use-toast";
import AppLayout from "@/components/layout/AppLayout";
import { useAuth } from '@/context/AuthContext';
import { Skeleton } from '@/components/ui/skeleton';
import { API_BASE_URL } from '@/config/api';

type SortDirection = 'ascending' | 'descending';
interface SortConfig {
  key: string;
  direction: SortDirection;
}

function VideoManagementContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { toast } = useToast();
  const { currentUser } = useAuth(); 
  
  const filterPatientIdParam = searchParams.get('patientId');
  const filterVideoIdParam = searchParams.get('videoId'); // For potential deep linking to a specific video

  const [videos, setVideos] = useState<Video[]>([]);
  const [patients, setPatients] = useState<Patient[]>([]); 
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [deletingVideoId, setDeletingVideoId] = useState<string | null>(null);
  const [sortConfig, setSortConfig] = useState<SortConfig | null>(null);

  const [isVideoPlayerOpen, setIsVideoPlayerOpen] = useState(false);
  const [currentVideoUrl, setCurrentVideoUrl] = useState<string | null>(null);
  const [currentVideoTitle, setCurrentVideoTitle] = useState<string | undefined>(undefined);


  const fetchVideosAndPatients = useCallback(async () => {
    if (!currentUser?.id) return;
    setIsLoading(true);
    setError(null);
    try {
      const [videosResponse, patientsResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/management/videos?admin_doctor_id=${currentUser.id}`),
        fetch(`${API_BASE_URL}/management/patients?admin_doctor_id=${currentUser.id}`) 
      ]);

      if (!videosResponse.ok) {
        const errData = await videosResponse.json();
        throw new Error(`Failed to fetch videos: ${videosResponse.status} ${errData.detail || videosResponse.statusText}`);
      }
      const videosData: Video[] = await videosResponse.json();
      setVideos(videosData.map(v => ({...v, id: String(v.id), patient_id: String(v.patient_id), action_id: v.action_id ? String(v.action_id) : null })));
      
      if (!patientsResponse.ok) {
        const errData = await patientsResponse.json();
        throw new Error(`Failed to fetch patients: ${patientsResponse.status} ${errData.detail || patientsResponse.statusText}`);
      }
      const patientsData: Patient[] = await patientsResponse.json();
      setPatients(patientsData.map(p => ({...p, id: String(p.id)})));

      if (filterVideoIdParam) {
        const videoToPlay = videosData.find(v => String(v.id) === filterVideoIdParam);
        if (videoToPlay) {
            handlePlayVideo(videoToPlay);
        }
      }


    } catch (e) {
      setError((e as Error).message);
      toast({ title: "Error", description: (e as Error).message, variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  }, [currentUser?.id, toast, filterVideoIdParam]);

  useEffect(() => {
    fetchVideosAndPatients();
  }, [fetchVideosAndPatients]);

  const handleSort = (key: string) => {
    let direction: SortDirection = 'ascending';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') {
      direction = 'descending';
    }
    setSortConfig({ key, direction });
  };
  
  const sortData = <T,>(data: T[], config: SortConfig | null, keyAccessor: (item: T, key: string) => any): T[] => {
    if (!config) return data;
    const { key, direction } = config;
    return [...data].sort((a, b) => {
      const aValue = keyAccessor(a, key);
      const bValue = keyAccessor(b, key);
  
      if (aValue === null || aValue === undefined) return 1;
      if (bValue === null || bValue === undefined) return -1;
  
      let comparison = 0;
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        comparison = aValue - bValue;
      } else if (typeof aValue === 'string' && typeof bValue === 'string') {
        comparison = aValue.toLowerCase().localeCompare(bValue.toLowerCase());
      } else {
        comparison = String(aValue).toLowerCase().localeCompare(String(bValue).toLowerCase());
      }
      return direction === 'ascending' ? comparison : -comparison;
    });
  };

  const displayedVideos = useMemo(() => {
    let filtered = videos;
    if (filterPatientIdParam) {
      filtered = filtered.filter(video => video.patient_id === filterPatientIdParam);
    }
    
    if (searchTerm) {
      const lowerSearchTerm = searchTerm.toLowerCase();
      filtered = filtered.filter(video =>
        (video.video_path?.toLowerCase() || '').includes(lowerSearchTerm) ||
        (video.patient_username?.toLowerCase() || '').includes(lowerSearchTerm) ||
        (video.action_id?.toString().toLowerCase() || '').includes(lowerSearchTerm)
      );
    }

    const videoKeyAccessor = (video: Video, key: string) => {
        switch(key) {
            case 'video_path': return video.video_path;
            case 'create_time': return video.create_time;
            case 'patient_username': return video.patient_username;
            case 'video_type': return video.original_video ? 'Original' : (video.inference_video ? 'Analysis' : 'Unknown');
            case 'action_id': return video.action_id ? parseInt(video.action_id) : null;
            default: return (video as any)[key];
        }
    }
    return sortData(filtered, sortConfig, videoKeyAccessor);
  }, [videos, filterPatientIdParam, searchTerm, sortConfig]);

  const patientForFilteredVideos = useMemo(() => {
    if (filterPatientIdParam) {
      return patients.find(p => p.id === filterPatientIdParam);
    }
    return null;
  }, [filterPatientIdParam, patients]);


  const handleDeleteVideo = (videoId: string) => setDeletingVideoId(videoId);

  const confirmDeleteVideo = async () => {
    if (!deletingVideoId || !currentUser?.id) return;
    const payload = {
        admin_doctor_id: parseInt(currentUser.id),
        video_id: parseInt(deletingVideoId),
        force: false 
    };
    try {
      const response = await fetch(`${API_BASE_URL}/management/video`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to delete video: ${response.statusText}`);
      }
      toast({ title: "Video Deleted", description: `Video has been deleted.`, variant: "destructive" });
      fetchVideosAndPatients(); 
    } catch (e) {
      toast({ title: "Error", description: (e as Error).message, variant: "destructive" });
    }
    setDeletingVideoId(null);
  };

  const handlePlayVideo = (video: Video) => {
    const videoTypeStr = video.original_video ? "original" : "inference";
    const videoUrl = `${API_BASE_URL}/videos/stream/${videoTypeStr}/${video.patient_id}/${video.id}`;
    setCurrentVideoUrl(videoUrl);
    setCurrentVideoTitle(video.video_path.split('/').pop() || `Video ID: ${video.id}`);
    setIsVideoPlayerOpen(true);
  };


  if (isLoading && !error) {
    return (
      <div className="flex flex-col gap-6">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">Video Management</h1>
        <Skeleton className="h-10 w-1/3 mb-4" /> 
        <div className="space-y-4">
          <Skeleton className="h-12 w-full" /> 
          <Skeleton className="h-64 w-full" /> 
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-destructive">
        <AlertCircle className="w-16 h-16" />
        <h2 className="text-2xl font-semibold">Error Loading Videos</h2>
        <p className="text-center">{error}</p>
        <Button onClick={() => { fetchVideosAndPatients(); }}>Retry</Button>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-3xl font-bold tracking-tight text-foreground">Video Management</h1>
      
      <div className="flex justify-between items-center mb-2">
          <div className="relative w-full sm:w-auto sm:max-w-xs">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Search videos..."
              className="pl-10 h-10 bg-card border-border focus:ring-primary"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
      </div>

      {filterPatientIdParam && patientForFilteredVideos && (
        <div className="mb-4 p-3 bg-primary/10 border border-primary/20 rounded-md flex justify-between items-center">
          <p className="text-sm text-primary font-medium">
            Showing videos for patient: {patientForFilteredVideos.username} (Case ID: {patientForFilteredVideos.case_id}).
          </p>
          <Button 
            variant="ghost" 
            size="sm"
            className="text-primary hover:bg-primary/20"
            onClick={() => router.push('/videos')}
          >
            Show All Videos
          </Button>
        </div>
      )}
      
      <VideosTable 
        videos={displayedVideos}
        onDelete={handleDeleteVideo}
        onPlayVideo={handlePlayVideo}
        sortConfig={sortConfig}
        onSort={handleSort}
        apiBaseUrl={API_BASE_URL}
      />

      <AlertDialog open={!!deletingVideoId} onOpenChange={(open) => !open && setDeletingVideoId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This action will mark the video as deleted. Associated analyses might also be affected.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setDeletingVideoId(null)}>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmDeleteVideo} className="bg-destructive hover:bg-destructive/90">Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <VideoPlayerModal 
        isOpen={isVideoPlayerOpen}
        onOpenChange={setIsVideoPlayerOpen}
        videoUrl={currentVideoUrl}
        videoTitle={currentVideoTitle}
      />
    </div>
  );
}

export default function VideoManagementPage() {
  return (
    <AppLayout>
      <Suspense fallback={<div className="flex justify-center items-center h-64 text-muted-foreground">Loading video data...</div>}>
        <VideoManagementContent />
      </Suspense>
    </AppLayout>
  );
}
