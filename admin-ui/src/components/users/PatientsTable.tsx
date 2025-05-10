
// @ts-nocheck
"use client";
import type { Patient } from "@/types"; 
import { useRouter } from "next/navigation";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { BriefcaseMedical, FilePenLine, Trash2, MoreHorizontal, ArrowUp, ArrowDown, ChevronsUpDown, Video } from "lucide-react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface SortConfig {
  key: string;
  direction: 'ascending' | 'descending';
}

interface PatientsTableProps {
  patients: Patient[];
  onEdit: (patient: Patient) => void;
  onDelete: (patientId: string) => void;
  sortConfig: SortConfig | null;
  onSort: (key: string) => void;
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


export function PatientsTable({ patients, onEdit, onDelete, sortConfig, onSort }: PatientsTableProps) {
  const router = useRouter();

  const handleDoctorClick = (doctorId?: string | null) => {
    if (doctorId) {
      router.push(`/users?tab=doctors&scrollToDoctorId=${doctorId}`);
    }
  };

  const handleVideoCountClick = (patientId: string) => {
    router.push(`/videos?patientId=${patientId}`);
  };
  
  const headers = [
    { key: 'username', label: 'Name', className: 'w-[180px]' },
    { key: 'age', label: 'Age', className: 'w-[80px]' },
    { key: 'gender', label: 'Gender', className: 'w-[100px]' },
    { key: 'case_id', label: 'Case ID / MRN', className: 'w-[180px]' },
    { key: 'attendingDoctorName', label: 'Attending Doctor', className: 'w-[200px]' },
    { key: 'notes', label: 'Notes', className: 'min-w-[150px]', sortable: false },
    { key: 'videoCount', label: 'Video Count', className: 'w-[130px] text-center' },
    { key: 'analysisCount', label: 'Analysis Count', className: 'w-[130px] text-center' },
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
            {patients.length === 0 ? (
              <TableRow>
                <TableCell colSpan={headers.length} className="h-24 text-center text-muted-foreground"> 
                  No patients found.
                </TableCell>
              </TableRow>
            ) : (
              patients.map((patient) => (
                <TableRow key={patient.id} className="hover:bg-muted/30 transition-colors">
                  <TableCell className="font-medium text-foreground">{patient.username}</TableCell>
                  <TableCell className="text-muted-foreground">{patient.age ?? '-'}</TableCell>
                  <TableCell className="text-muted-foreground">{patient.gender ?? '-'}</TableCell>
                  <TableCell className="text-muted-foreground font-mono text-sm">{patient.case_id}</TableCell>
                  <TableCell>
                    <Button 
                      variant="link" 
                      className="p-0 h-auto text-primary hover:underline font-medium" 
                      onClick={() => handleDoctorClick(patient.doctor_id)}
                      disabled={!patient.doctor_id || patient.doctor_id === ''}
                    >
                      {patient.attendingDoctorName || 'N/A'} 
                      {patient.doctor_id && patient.doctor_id !== '' && <BriefcaseMedical className="ml-1.5 h-4 w-4 inline-block" />}
                    </Button>
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground max-w-[150px] truncate">
                    {patient.notes ? (
                      <Tooltip delayDuration={100}>
                        <TooltipTrigger asChild>
                           <span>{patient.notes}</span>
                        </TooltipTrigger>
                        <TooltipContent side="top" align="start" className="max-w-xs bg-background border-border text-foreground shadow-lg p-2 rounded-md">
                          <p className="text-xs whitespace-pre-wrap">{patient.notes}</p>
                        </TooltipContent>
                      </Tooltip>
                    ) : (
                      '-'
                    )}
                  </TableCell>
                  <TableCell className="text-center">
                     <Button 
                      variant="link" 
                      className="p-0 h-auto text-primary hover:underline font-medium" 
                      onClick={() => handleVideoCountClick(patient.id)}
                      disabled={patient.videoCount === undefined || patient.videoCount === null}
                    >
                      {patient.videoCount ?? 'N/A'} <Video className="ml-1.5 h-4 w-4 inline-block" />
                    </Button>
                  </TableCell>
                  <TableCell className="text-center text-muted-foreground">{patient.analysisCount ?? 'N/A'}</TableCell>
                  <TableCell className="text-right">
                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" className="h-8 w-8 p-0">
                            <span className="sr-only">Open menu</span>
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => onEdit(patient)}>
                            <FilePenLine className="mr-2 h-4 w-4" /> Edit
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => onDelete(patient.id)} className="text-destructive focus:text-destructive focus:bg-destructive/10">
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
