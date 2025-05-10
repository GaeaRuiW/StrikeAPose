
// @ts-nocheck
"use client";

import React from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import type { Patient, Doctor } from '@/types'; 
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";

const patientFormSchema = z.object({
  username: z.string().min(1, "Name is required.").min(2, { message: "Name must be at least 2 characters." }),
  age: z.coerce.number().min(0, "Age cannot be negative.").optional().nullable(),
  gender: z.enum(['Male', 'Female', 'Other']).optional().nullable(),
  case_id: z.string().min(1, "Case ID is required.").min(3, { message: "Case ID/MRN must be at least 3 characters." }),
  doctor_id: z.string().optional().nullable(), // Can be actual ID string, or null
  notes: z.string().optional().nullable(),
});

export type PatientFormData = z.infer<typeof patientFormSchema>;

interface PatientFormDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (data: PatientFormData) => void;
  defaultValues?: Patient | null; 
  doctors: Doctor[]; 
}

const NO_DOCTOR_VALUE = "__NONE__"; // Unique value for "None" option

export function PatientFormDialog({ open, onOpenChange, onSubmit, defaultValues, doctors }: PatientFormDialogProps) {
  const form = useForm<PatientFormData>({
    resolver: zodResolver(patientFormSchema),
    defaultValues: {
      username: defaultValues?.username || "",
      age: defaultValues?.age ?? undefined,
      gender: defaultValues?.gender || undefined,
      case_id: defaultValues?.case_id || "",
      doctor_id: defaultValues?.doctor_id ?? null, // Use null for no doctor, or actual ID
      notes: defaultValues?.notes || "",
    },
  });

  const handleSubmit = (data: PatientFormData) => {
    const submissionData = {
      ...data,
      age: data.age === undefined || data.age === null || isNaN(data.age) ? null : Number(data.age),
      gender: data.gender === undefined || data.gender === null ? null : data.gender,
      doctor_id: data.doctor_id, // Will be null if "None" was selected, or actual ID
      notes: data.notes === "" || data.notes === undefined || data.notes === null ? null : data.notes,
    };
    onSubmit(submissionData as PatientFormData);
    form.reset({ 
        username: "", 
        age: undefined, 
        gender: undefined, 
        case_id: "", 
        doctor_id: null, // Reset to null for unassigned
        notes: "" 
    });
  };
  
  React.useEffect(() => {
    if (open) { // Reset form when dialog opens or defaultValues change
      form.reset({
        username: defaultValues?.username || "",
        age: defaultValues?.age ?? undefined,
        gender: defaultValues?.gender || undefined,
        case_id: defaultValues?.case_id || "",
        doctor_id: defaultValues?.doctor_id ?? null, // Use null if no doctor_id
        notes: defaultValues?.notes || "",
      });
    }
  }, [defaultValues, form, open]);


  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[480px] bg-card text-card-foreground">
        <DialogHeader>
          <DialogTitle>{defaultValues ? 'Edit Patient' : 'Add New Patient'}</DialogTitle>
          <DialogDescription>
            {defaultValues ? 'Update the details for this patient.' : 'Enter the details for the new patient.'}
          </DialogDescription>
        </DialogHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(handleSubmit)} className="grid gap-4 py-4">
            <FormField
              control={form.control}
              name="username"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Patient Name <span className="text-destructive">*</span></FormLabel>
                  <FormControl>
                    <Input placeholder="Jane Roe" {...field} className="bg-background border-input"/>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="age"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Age</FormLabel>
                    <FormControl>
                      <Input type="number" placeholder="30" {...field} onChange={e => field.onChange(e.target.value === '' ? null : +e.target.value)} value={field.value ?? ''} className="bg-background border-input"/>
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="gender"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Gender</FormLabel>
                    <Select onValueChange={field.onChange} value={field.value ?? undefined} defaultValue={field.value ?? undefined}>
                      <FormControl>
                        <SelectTrigger className="bg-background border-input">
                          <SelectValue placeholder="Select gender" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="Male">Male</SelectItem>
                        <SelectItem value="Female">Female</SelectItem>
                        <SelectItem value="Other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
            <FormField
              control={form.control}
              name="case_id"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Case ID / Medical Record Number <span className="text-destructive">*</span></FormLabel>
                  <FormControl>
                    <Input placeholder="MRN12345" {...field} className="bg-background border-input"/>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="doctor_id"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Attending Doctor</FormLabel>
                  <Select 
                    onValueChange={(value) => field.onChange(value === NO_DOCTOR_VALUE ? null : value)} 
                    value={field.value === null ? NO_DOCTOR_VALUE : (field.value || "")} // Map null to NO_DOCTOR_VALUE for select state, "" for placeholder
                  >
                    <FormControl>
                      <SelectTrigger className="bg-background border-input">
                        <SelectValue placeholder="Select doctor" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value={NO_DOCTOR_VALUE}><em>None</em></SelectItem>
                      {doctors.map(doctor => (
                        <SelectItem key={doctor.id} value={doctor.id.toString()}>
                          {doctor.username} - {doctor.department}
                        </SelectItem>
                      ))}
                       {doctors.length === 0 && !defaultValues?.doctor_id && <SelectItem value="no_doctors_available_placeholder" disabled>No doctors available</SelectItem>}
                    </SelectContent>
                  </Select>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="notes"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Notes</FormLabel>
                  <FormControl>
                    <Textarea placeholder="Optional notes about the patient..." {...field} value={field.value ?? ''} className="bg-background border-input"/>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => { form.reset(); onOpenChange(false); }}>Cancel</Button>
              <Button type="submit">{defaultValues ? 'Save Changes' : 'Add Patient'}</Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}

