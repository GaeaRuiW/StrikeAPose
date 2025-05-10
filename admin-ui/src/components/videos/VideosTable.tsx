
// @ts-nocheck
"use client";
import type { Video } from "@/types";
import Image from 'next/image';
import Link from 'next/link';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Trash2, MoreHorizontal, ArrowUp, ArrowDown, ChevronsUpDown, ExternalLink, PlayCircle } from "lucide-react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Badge } from "@/components/ui/badge";
import { format } from 'date-fns';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";


interface SortConfig {
  key: string;
  direction: 'ascending' | 'descending';
}

interface VideosTableProps {
  videos: Video[];
  onDelete: (videoId: string) => void;
  onPlayVideo: (video: Video) => void;
  sortConfig: SortConfig | null;
  onSort: (key: string) => void;
  apiBaseUrl: string;
}

const SortableHeader = ({ children, sortKey, currentSort, onSort }: { children: React.ReactNode, sortKey: string, currentSort: SortConfig | null, onSort: (key: string) => void }) => {
  const isSorted = currentSort?.key === sortKey;
  const Icon = isSorted ? (currentSort.direction === 'ascending' ? ArrowUp : ArrowDown) : ChevronsUpDown;
  return (
    <Button variant="ghost" onClick={() => onSort(sortKey)} className="px-1 py-0.5 h-auto font-medium text-muted-foreground hover:text-foreground group">
      {children}
      <Icon className={`ml-2 h-4 w-4 ${isSorted ? 'text-foreground' : 'opacity-50 group-hover:opacity-100'}`} />
    </Button>
  );
};

const getVideoTypeString = (video: Video): string => {
  if (video.original_video) return "original";
  if (video.inference_video) return "inference";
  return "unknown";
}

const getVideoTypeDisplay = (video: Video): string => {
  if (video.original_video) return "Original";
  if (video.inference_video) return "Analysis";
  return "Unknown";
};

const getThumbnailUrl = (video: Video, apiBaseUrl: string): string => {
  const typeString = getVideoTypeString(video);
  if (typeString === "unknown") return "https://picsum.photos/seed/placeholder/100/60"; // Fallback
  return `${apiBaseUrl}/videos/thumbnail_image/${typeString}/${video.patient_id}/${video.id}`;
};


export function VideosTable({ videos, onDelete, onPlayVideo, sortConfig, onSort, apiBaseUrl }: VideosTableProps) {
  
  const headers = [
    { key: 'thumbnail', label: 'Thumbnail', className: 'w-[120px]', sortable: false },
    { key: 'video_path', label: 'Video Info', className: 'w-[250px]' },
    { key: 'patient_username', label: 'Patient', className: 'w-[180px]' },
    { key: 'create_time', label: 'Upload Date', className: 'w-[180px]' },
    { key: 'action_id', label: 'Analysis ID', className: 'w-[150px]' },
    { key: 'actions', label: 'Actions', className: 'w-[80px] text-right', sortable: false },
  ];
  
  return (
    <TooltipProvider>
    <div className="rounded-lg border shadow-md overflow-x-auto bg-card">
      <Table>
        <TableHeader>
          <TableRow className="bg-muted/50 hover:bg-muted/60">
            {headers.map(header => (
              <TableHead key={header.key} className={header.className}>
                {header.sortable !== false ? (
                  <SortableHeader sortKey={header.key} currentSort={sortConfig} onSort={onSort}>
                    {header.label}
                  </SortableHeader>
                ) : (
                  header.label
                )}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {videos.length === 0 ? (
            <TableRow>
              <TableCell colSpan={headers.length} className="h-24 text-center text-muted-foreground">
                No videos found.
              </TableCell>
            </TableRow>
          ) : (
            videos.map((video) => (
              <TableRow key={video.id} className="hover:bg-muted/30 transition-colors">
                <TableCell className="p-2">
                  <button 
                    onClick={() => onPlayVideo(video)} 
                    className="relative block w-[100px] h-[60px] rounded overflow-hidden group cursor-pointer"
                    aria-label={`Play video ${video.id}`}
                  >
                    <Image 
                      src={getThumbnailUrl(video, apiBaseUrl)} 
                      alt={`Thumbnail for video ${video.id}`} 
                      width={100} 
                      height={60} 
                      className="object-cover w-full h-full"
                      unoptimized 
                      onError={(e) => { e.currentTarget.src = 'https://picsum.photos/seed/error/100/60'; }} 
                      data-ai-hint="video thumbnail"
                    />
                    <div className="absolute inset-0 bg-black/30 group-hover:bg-black/50 flex items-center justify-center transition-opacity opacity-0 group-hover:opacity-100">
                        <PlayCircle className="h-8 w-8 text-white/80" />
                    </div>
                  </button>
                </TableCell>
                <TableCell>
                  <Tooltip delayDuration={100}>
                    <TooltipTrigger asChild>
                        <div className="font-medium text-foreground truncate max-w-[200px]" title={video.video_path}>
                            {video.video_path.split('/').pop() || video.video_path}
                        </div>
                    </TooltipTrigger>
                    <TooltipContent side="top" align="start">
                      <p className="text-xs">{video.video_path}</p>
                    </TooltipContent>
                  </Tooltip>
                  <Badge variant={video.original_video ? "secondary" : "outline"} className="mt-1">
                    {getVideoTypeDisplay(video)}
                  </Badge>
                </TableCell>
                <TableCell className="text-muted-foreground">
                    <Link href={`/users?tab=patients&patientId=${video.patient_id}`} className="hover:underline text-primary">
                        {video.patient_username || 'N/A'}
                    </Link>
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {format(new Date(video.create_time), "MMM dd, yyyy HH:mm")}
                </TableCell>
                <TableCell className="text-muted-foreground font-mono text-sm">
                  {video.action_id ? (
                    <Link href={`/analyses?analysisId=${video.action_id}`} className="hover:underline text-primary">
                      {video.action_id}
                    </Link>
                  ) : (
                    '-'
                  )}
                </TableCell>
                <TableCell className="text-right">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" className="h-8 w-8 p-0">
                        <span className="sr-only">Open menu</span>
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => onPlayVideo(video)}>
                        <PlayCircle className="mr-2 h-4 w-4" /> Play Video
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => window.open(`${apiBaseUrl}/videos/video/${getVideoTypeString(video)}/${video.patient_id}/${video.id}`, '_blank')}>
                        <ExternalLink className="mr-2 h-4 w-4" /> View Raw Video
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => onDelete(video.id)} className="text-destructive focus:text-destructive focus:bg-destructive/10">
                        <Trash2 className="mr-2 h-4 w-4" /> Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
    </TooltipProvider>
  );
}

