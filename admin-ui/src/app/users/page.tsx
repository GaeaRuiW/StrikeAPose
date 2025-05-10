
// @ts-nocheck
"use client";

import React, { useEffect, useMemo, useState, Suspense, useCallback } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import type { Doctor, Patient } from '@/types'; // Use updated types
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DoctorsTable } from '@/components/users/DoctorsTable';
import { PatientsTable } from '@/components/users/PatientsTable';
import { Input } from '@/components/ui/input';
import { Search, PlusCircle, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { DoctorFormDialog, DoctorFormData } from '@/components/users/DoctorFormDialog';
import { PatientFormDialog, PatientFormData } from '@/components/users/PatientFormDialog';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { useToast } from "@/hooks/use-toast";
import AppLayout from "@/components/layout/AppLayout";
import { useAuth } from '@/context/AuthContext'; // For admin_doctor_id
import { Skeleton } from '@/components/ui/skeleton';

import { API_BASE_URL } from '@/config/api';

type SortDirection = 'ascending' | 'descending';
interface SortConfig {
  key: string;
  direction: SortDirection;
}

// Mapping frontend permission strings to backend role_id
const permissionToRoleId = {
  'Admin': 1,
  'Doctor': 2,
};
const roleIdToPermissionString = (roleId?: number | null): string => {
  if (roleId === 1) return 'Admin';
  if (roleId === 2) return 'Doctor';
  return 'Unknown';
};


function UserManagementContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { toast } = useToast();
  const { currentUser } = useAuth(); 
  
  const activeTab = searchParams.get('tab') || 'doctors';
  const filterDoctorIdParam = searchParams.get('doctorId');
  const scrollToDoctorIdParam = searchParams.get('scrollToDoctorId');

  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [isLoading, setIsLoading] = useState({ doctors: true, patients: true });
  const [error, setError] = useState<string | null>(null);

  const [searchTermDoctors, setSearchTermDoctors] = useState('');
  const [searchTermPatients, setSearchTermPatients] = useState('');

  const [isDoctorModalOpen, setIsDoctorModalOpen] = useState(false);
  const [editingDoctor, setEditingDoctor] = useState<Doctor | null>(null);
  const [isPatientModalOpen, setIsPatientModalOpen] = useState(false);
  const [editingPatient, setEditingPatient] = useState<Patient | null>(null);

  const [deletingDoctorId, setDeletingDoctorId] = useState<string | null>(null);
  const [deletingPatientId, setDeletingPatientId] = useState<string | null>(null);
  const [assignDoctorIdForDeletion, setAssignDoctorIdForDeletion] = useState<string | null>(null);

  const [sortConfigDoctors, setSortConfigDoctors] = useState<SortConfig | null>(null);
  const [sortConfigPatients, setSortConfigPatients] = useState<SortConfig | null>(null);


  const fetchDoctors = useCallback(async () => {
    if (!currentUser?.id) return;
    setIsLoading(prev => ({ ...prev, doctors: true }));
    try {
      const response = await fetch(`${API_BASE_URL}/management/doctors?admin_doctor_id=${currentUser.id}`);
      if (!response.ok) throw new Error(`Failed to fetch doctors: ${response.statusText}`);
      const data: Doctor[] = await response.json();
      setDoctors(data.map(d => ({...d, id: String(d.id)}))); 
    } catch (e) {
      setError((e as Error).message);
      toast({ title: "Error", description: "Could not fetch doctors.", variant: "destructive" });
    } finally {
      setIsLoading(prev => ({ ...prev, doctors: false }));
    }
  }, [currentUser?.id, toast]);

  const fetchPatients = useCallback(async () => {
    if (!currentUser?.id) return;
    setIsLoading(prev => ({ ...prev, patients: true }));
    try {
      const response = await fetch(`${API_BASE_URL}/management/patients?admin_doctor_id=${currentUser.id}`);
      if (!response.ok) throw new Error(`Failed to fetch patients: ${response.statusText}`);
      const data: Patient[] = await response.json();
      setPatients(data.map(p => ({...p, id: String(p.id), doctor_id: p.doctor_id ? String(p.doctor_id) : null}))); 
    } catch (e) {
      setError((e as Error).message);
      toast({ title: "Error", description: "Could not fetch patients.", variant: "destructive" });
    } finally {
      setIsLoading(prev => ({ ...prev, patients: false }));
    }
  }, [currentUser?.id, toast]);

  useEffect(() => {
    fetchDoctors();
    fetchPatients();
  }, [fetchDoctors, fetchPatients]);

  const handleSortDoctors = (key: string) => {
    let direction: SortDirection = 'ascending';
    if (sortConfigDoctors && sortConfigDoctors.key === key && sortConfigDoctors.direction === 'ascending') {
      direction = 'descending';
    }
    setSortConfigDoctors({ key, direction });
  };

  const handleSortPatients = (key: string) => {
    let direction: SortDirection = 'ascending';
    if (sortConfigPatients && sortConfigPatients.key === key && sortConfigPatients.direction === 'ascending') {
      direction = 'descending';
    }
    setSortConfigPatients({ key, direction });
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
        // Fallback for mixed types or other types
        comparison = String(aValue).toLowerCase().localeCompare(String(bValue).toLowerCase());
      }
      return direction === 'ascending' ? comparison : -comparison;
    });
  };


  const displayedDoctors = useMemo(() => {
    let filteredDoctors = doctors.filter(doctor =>
      (doctor.username?.toLowerCase() || '').includes(searchTermDoctors.toLowerCase()) ||
      (doctor.email?.toLowerCase() || '').includes(searchTermDoctors.toLowerCase()) ||
      (doctor.department?.toLowerCase() || '').includes(searchTermDoctors.toLowerCase())
    );

    const doctorKeyAccessor = (doctor: Doctor, key: string) => {
        switch(key) {
            case 'username': return doctor.username;
            case 'email': return doctor.email;
            case 'phone': return doctor.phone;
            case 'department': return doctor.department;
            case 'patientCount': return doctor.patientCount;
            case 'role_id': return roleIdToPermissionString(doctor.role_id);
            default: return (doctor as any)[key];
        }
    }
    return sortData(filteredDoctors, sortConfigDoctors, doctorKeyAccessor);
  }, [doctors, searchTermDoctors, sortConfigDoctors]);


  const displayedPatients = useMemo(() => {
    let filtered = patients;
    if (activeTab === 'patients' && filterDoctorIdParam) {
      filtered = filtered.filter(patient => patient.doctor_id === filterDoctorIdParam);
    }
    filtered = filtered.filter(patient =>
      (patient.username?.toLowerCase() || '').includes(searchTermPatients.toLowerCase()) ||
      (patient.case_id?.toLowerCase() || '').includes(searchTermPatients.toLowerCase()) ||
      (patient.attendingDoctorName && patient.attendingDoctorName.toLowerCase().includes(searchTermPatients.toLowerCase()))
    );
    const patientKeyAccessor = (patient: Patient, key: string) => {
        switch(key) {
            case 'username': return patient.username;
            case 'age': return patient.age;
            case 'gender': return patient.gender;
            case 'case_id': return patient.case_id;
            case 'attendingDoctorName': return patient.attendingDoctorName;
            case 'videoCount': return patient.videoCount;
            case 'analysisCount': return patient.analysisCount;
            default: return (patient as any)[key];
        }
    }
    return sortData(filtered, sortConfigPatients, patientKeyAccessor);
  }, [patients, filterDoctorIdParam, searchTermPatients, activeTab, sortConfigPatients]);


  const handleTabChange = (newTabValue: string) => {
    const newParams = new URLSearchParams(searchParams.toString());
    newParams.set('tab', newTabValue);
    if (newTabValue === 'doctors' && !newParams.has('scrollToDoctorId')) newParams.delete('doctorId');
    if (newTabValue === 'patients') newParams.delete('scrollToDoctorId');
    router.push(`/users?${newParams.toString()}`, { scroll: false });
  };
  
  useEffect(() => {
    const currentParams = new URLSearchParams(searchParams.toString());
    let paramsChanged = false;
    if (activeTab === 'doctors' && filterDoctorIdParam && !scrollToDoctorIdParam) {
      currentParams.delete('doctorId');
      paramsChanged = true;
    }
    if (activeTab === 'patients' && scrollToDoctorIdParam) {
      currentParams.delete('scrollToDoctorId');
      paramsChanged = true;
    }
    if (paramsChanged) router.replace(`/users?${currentParams.toString()}`, { scroll: false });
  }, [activeTab, filterDoctorIdParam, scrollToDoctorIdParam, searchParams, router]);

  const doctorForFilteredPatients = useMemo(() => {
    if (activeTab === 'patients' && filterDoctorIdParam) {
      return doctors.find(d => d.id === filterDoctorIdParam);
    }
    return null;
  }, [activeTab, filterDoctorIdParam, doctors]);

  // Doctor CRUD
  const handleAddDoctor = () => {
    setEditingDoctor(null);
    setIsDoctorModalOpen(true);
  };

  const handleEditDoctor = (doctor: Doctor) => {
    setEditingDoctor(doctor);
    setIsDoctorModalOpen(true);
  };

  const handleSaveDoctor = async (data: DoctorFormData) => {
    if (!currentUser?.id) return;
    const payload = { ...data, admin_doctor_id: parseInt(currentUser.id) };
    
    try {
      let response;
      if (editingDoctor) { // Update
        response = await fetch(`${API_BASE_URL}/management/doctor`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...payload, doctor_id: parseInt(editingDoctor.id) }),
        });
      } else { // Create
        response = await fetch(`${API_BASE_URL}/management/doctor`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to save doctor: ${response.statusText}`);
      }
      toast({ title: editingDoctor ? "Doctor Updated" : "Doctor Added", description: `${data.username} has been saved successfully.` });
      fetchDoctors(); 
    } catch (e) {
      toast({ title: "Error", description: (e as Error).message, variant: "destructive" });
    }
    setIsDoctorModalOpen(false);
  };

  const handleDeleteDoctor = (doctorId: string) => {
    const doctorToDelete = doctors.find(d => d.id === doctorId);
    if (!doctorToDelete) return;

    if (doctorToDelete.patientCount && doctorToDelete.patientCount > 0) {
        const otherDoctors = doctors.filter(d => d.id !== doctorId);
        if (otherDoctors.length === 0) {
            toast({ title: "Cannot Delete Doctor", description: "This doctor has patients and there are no other doctors to reassign them to.", variant: "destructive"});
            return;
        }
        setAssignDoctorIdForDeletion(otherDoctors[0].id); 
        setDeletingDoctorId(doctorId); 
    } else {
        let assignId = currentUser?.id && currentUser.id !== doctorId ? currentUser.id : (doctors.find(d => d.id !== doctorId)?.id || doctorId);
        setAssignDoctorIdForDeletion(assignId);
        setDeletingDoctorId(doctorId);
    }
  };


  const confirmDeleteDoctor = async () => {
    if (!deletingDoctorId || !currentUser?.id || !assignDoctorIdForDeletion) return;
    
    const payload = {
      admin_doctor_id: parseInt(currentUser.id),
      doctor_id: parseInt(deletingDoctorId),
      assign_doctor_id: parseInt(assignDoctorIdForDeletion)
    };

    try {
      const response = await fetch(`${API_BASE_URL}/management/doctor`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to delete doctor: ${response.statusText}`);
      }
      toast({ title: "Doctor Deleted", description: `Doctor has been deleted.`, variant: "destructive"});
      fetchDoctors(); 
      fetchPatients(); 
    } catch (e) {
      toast({ title: "Error", description: (e as Error).message, variant: "destructive" });
    }
    setDeletingDoctorId(null);
    setAssignDoctorIdForDeletion(null);
  };

  // Patient CRUD
  const handleAddPatient = () => {
    setEditingPatient(null);
    setIsPatientModalOpen(true);
  };

  const handleEditPatient = (patient: Patient) => {
    setEditingPatient(patient);
    setIsPatientModalOpen(true);
  };

  const handleSavePatient = async (data: PatientFormData) => {
    if (!currentUser?.id) return;
    const payload = { 
      ...data, 
      admin_doctor_id: parseInt(currentUser.id), 
      doctor_id: data.doctor_id ? parseInt(data.doctor_id) : null 
    };
    
    try {
      let response;
      if (editingPatient) { // Update
        response = await fetch(`${API_BASE_URL}/management/patient`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...payload, patient_id: parseInt(editingPatient.id) }),
        });
      } else { // Create
        response = await fetch(`${API_BASE_URL}/management/patient`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      }
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to save patient: ${response.statusText}`);
      }
      toast({ title: editingPatient ? "Patient Updated" : "Patient Added", description: `${data.username} has been saved successfully.` });
      fetchPatients(); 
      fetchDoctors(); 
    } catch (e) {
      toast({ title: "Error", description: (e as Error).message, variant: "destructive" });
    }
    setIsPatientModalOpen(false);
  };

  const handleDeletePatient = (patientId: string) => setDeletingPatientId(patientId);

  const confirmDeletePatient = async () => {
    if (!deletingPatientId || !currentUser?.id) return;
    const payload = {
        admin_doctor_id: parseInt(currentUser.id),
        patient_id: parseInt(deletingPatientId),
        force: false 
    };
    try {
      const response = await fetch(`${API_BASE_URL}/management/patient`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to delete patient: ${response.statusText}`);
      }
      toast({ title: "Patient Deleted", description: `Patient has been deleted.`, variant: "destructive" });
      fetchPatients(); 
      fetchDoctors(); 
    } catch (e) {
      toast({ title: "Error", description: (e as Error).message, variant: "destructive" });
    }
    setDeletingPatientId(null);
  };

  if (isLoading.doctors || isLoading.patients && !error) {
    return (
      <div className="flex flex-col gap-6">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">User Management</h1>
        <Skeleton className="h-10 w-1/4" /> 
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
        <h2 className="text-2xl font-semibold">Error Loading Data</h2>
        <p className="text-center">{error}</p>
        <Button onClick={() => { fetchDoctors(); fetchPatients(); setError(null); }}>Retry</Button>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-3xl font-bold tracking-tight text-foreground">User Management</h1>
      <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
          <TabsList className="bg-muted p-1 rounded-lg">
            <TabsTrigger value="doctors" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">Doctors</TabsTrigger>
            <TabsTrigger value="patients" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">Patients</TabsTrigger>
          </TabsList>
          
          <div className="flex flex-col sm:flex-row gap-2 w-full sm:w-auto">
            {activeTab === 'doctors' && (
              <>
                <div className="relative w-full sm:w-auto sm:max-w-xs">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    type="search"
                    placeholder="Search doctors..."
                    className="pl-10 h-10 bg-card border-border focus:ring-primary"
                    value={searchTermDoctors}
                    onChange={(e) => setSearchTermDoctors(e.target.value)}
                  />
                </div>
                <Button onClick={handleAddDoctor} className="w-full sm:w-auto">
                  <PlusCircle className="mr-2 h-4 w-4" /> Add Doctor
                </Button>
              </>
            )}
            {activeTab === 'patients' && (
              <>
                <div className="relative w-full sm:w-auto sm:max-w-xs">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    type="search"
                    placeholder="Search patients..."
                    className="pl-10 h-10 bg-card border-border focus:ring-primary"
                    value={searchTermPatients}
                    onChange={(e) => setSearchTermPatients(e.target.value)}
                  />
                </div>
                <Button onClick={handleAddPatient} className="w-full sm:w-auto">
                  <PlusCircle className="mr-2 h-4 w-4" /> Add Patient
                </Button>
              </>
            )}
          </div>
        </div>

        {activeTab === 'patients' && doctorForFilteredPatients && (
          <div className="mb-4 p-3 bg-primary/10 border border-primary/20 rounded-md flex justify-between items-center">
            <p className="text-sm text-primary font-medium">
              Showing patients for Dr. {doctorForFilteredPatients.username}.
            </p>
            <Button 
              variant="ghost" 
              size="sm"
              className="text-primary hover:bg-primary/20"
              onClick={() => router.push('/users?tab=patients')}
            >
              Show All Patients
            </Button>
          </div>
        )}

        <TabsContent value="doctors">
          <DoctorsTable 
            doctors={displayedDoctors} 
            scrollToDoctorId={scrollToDoctorIdParam}
            onEdit={handleEditDoctor}
            onDelete={handleDeleteDoctor}
            sortConfig={sortConfigDoctors}
            onSort={handleSortDoctors}
          />
        </TabsContent>
        <TabsContent value="patients">
          <PatientsTable 
            patients={displayedPatients} 
            onEdit={handleEditPatient}
            onDelete={handleDeletePatient}
            sortConfig={sortConfigPatients}
            onSort={handleSortPatients}
          />
        </TabsContent>
      </Tabs>

      {isDoctorModalOpen && (
        <DoctorFormDialog
          open={isDoctorModalOpen}
          onOpenChange={setIsDoctorModalOpen}
          onSubmit={handleSaveDoctor}
          defaultValues={editingDoctor}
        />
      )}

      {isPatientModalOpen && (
         <PatientFormDialog
            open={isPatientModalOpen}
            onOpenChange={setIsPatientModalOpen}
            onSubmit={handleSavePatient}
            defaultValues={editingPatient}
            doctors={doctors} 
        />
      )}

      <AlertDialog open={!!deletingDoctorId} onOpenChange={(open) => !open && setDeletingDoctorId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This action cannot be undone. This will permanently delete Dr. {doctors.find(d => d.id === deletingDoctorId)?.username}. 
              {doctors.find(d => d.id === deletingDoctorId)?.patientCount ?? 0 > 0 ? ` Patients will be reassigned to Dr. ${doctors.find(d => d.id === assignDoctorIdForDeletion)?.username || 'another doctor'}.` : ''}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setDeletingDoctorId(null)}>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmDeleteDoctor} className="bg-destructive hover:bg-destructive/90">Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={!!deletingPatientId} onOpenChange={(open) => !open && setDeletingPatientId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This action cannot be undone. This will permanently delete patient {patients.find(p => p.id === deletingPatientId)?.username}.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setDeletingPatientId(null)}>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmDeletePatient} className="bg-destructive hover:bg-destructive/90">Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

    </div>
  );
}

export default function UserManagementPage() {
  return (
    <AppLayout>
      <Suspense fallback={<div className="flex justify-center items-center h-64 text-muted-foreground">Loading user data...</div>}>
        <UserManagementContent />
      </Suspense>
    </AppLayout>
  );
}

