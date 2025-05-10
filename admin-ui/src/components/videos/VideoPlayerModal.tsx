
"use client";

import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { X } from 'lucide-react';

interface VideoPlayerModalProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  videoUrl: string | null;
  videoTitle?: string;
}

export function VideoPlayerModal({ isOpen, onOpenChange, videoUrl, videoTitle }: VideoPlayerModalProps) {
  if (!videoUrl) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[800px] p-0 bg-black border-border shadow-2xl rounded-lg overflow-hidden">
        <DialogHeader className="p-4 bg-background/10 flex flex-row justify-between items-center">
          <DialogTitle className="text-primary-foreground">{videoTitle || 'Video Player'}</DialogTitle>
          <DialogClose asChild>
            <Button variant="ghost" size="icon" className="text-primary-foreground hover:bg-background/20">
              <X className="h-5 w-5" />
              <span className="sr-only">Close</span>
            </Button>
          </DialogClose>
        </DialogHeader>
        <div className="aspect-video w-full">
          <video src={videoUrl} controls autoPlay className="w-full h-full" />
        </div>
      </DialogContent>
    </Dialog>
  );
}
