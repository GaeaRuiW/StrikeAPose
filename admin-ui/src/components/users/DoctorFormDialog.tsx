
// @ts-nocheck
"use client";

import React from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import type { Doctor } from '@/types';
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
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";

const permissionToRoleId = {
  'Admin': 1,
  'Doctor': 2,
};

const NO_ROLE_VALUE = "__NO_ROLE__"; // Unique value for "None" option for Role

// Schema factory to handle conditional password requirement
const createDoctorFormSchema = (isEditing: boolean) => z.object({
  username: z.string().min(1, "Username is required.").min(2, { message: "Username must be at least 2 characters." }),
  email: z.string().min(1, "Email is required.").email({ message: "Invalid email address." }),
  password: isEditing
    ? z.string().min(6, { message: "New password must be at least 6 characters." }).optional().or(z.literal(''))
    : z.string().min(6, { message: "Password must be at least 6 characters." }),
  phone: z.string().optional().or(z.literal('')),
  department: z.string().optional().or(z.literal('')),
  role_id: z.union([z.nativeEnum(permissionToRoleId), z.number().min(1), z.string().regex(/^\d+$/).transform(Number)])
           .optional()
           .nullable(),
  notes: z.string().optional().nullable(),
});

export type DoctorFormData = z.infer<ReturnType<typeof createDoctorFormSchema>>;

interface DoctorFormDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (data: DoctorFormData) => void;
  defaultValues?: Doctor | null;
}

export function DoctorFormDialog({ open, onOpenChange, onSubmit, defaultValues }: DoctorFormDialogProps) {
  const isEditing = !!defaultValues;
  const doctorFormSchema = createDoctorFormSchema(isEditing);

  const form = useForm<DoctorFormData>({
    resolver: zodResolver(doctorFormSchema),
    defaultValues: {
      username: defaultValues?.username || "",
      email: defaultValues?.email || "",
      password: "", // Always blank for edit, or for new
      phone: defaultValues?.phone || "",
      department: defaultValues?.department || "",
      role_id: defaultValues?.role_id ?? null, 
      notes: defaultValues?.notes || "",
    },
  });

  const handleSubmit = (data: DoctorFormData) => {
    const submissionData = { ...data };
    if (isEditing && !data.password) { // If editing and password field is empty, don't send it
      delete submissionData.password;
    }
    // Ensure optional fields are correctly formatted as null if empty
    submissionData.phone = data.phone || null;
    submissionData.department = data.department || null;
    submissionData.role_id = data.role_id ? Number(data.role_id) : null;
    submissionData.notes = data.notes || null;
    
    onSubmit(submissionData);
    form.reset({ 
        username: "",
        email: "",
        password: "",
        phone: "",
        department: "",
        role_id: null,
        notes: ""
    });
  };

  React.useEffect(() => {
    if (open) { 
        form.reset({
            username: defaultValues?.username || "",
            email: defaultValues?.email || "",
            password: "",
            phone: defaultValues?.phone || "",
            department: defaultValues?.department || "",
            role_id: defaultValues?.role_id ?? null,
            notes: defaultValues?.notes || "",
        });
    }
  }, [defaultValues, form, open]);


  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[480px] bg-card text-card-foreground">
        <DialogHeader>
          <DialogTitle>{defaultValues ? 'Edit Doctor' : 'Add New Doctor'}</DialogTitle>
          <DialogDescription>
            {defaultValues ? 'Update the details for this doctor.' : 'Enter the details for the new doctor.'}
          </DialogDescription>
        </DialogHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(handleSubmit)} className="grid gap-4 py-4">
            <FormField
              control={form.control}
              name="username"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Username <span className="text-destructive">*</span></FormLabel>
                  <FormControl>
                    <Input placeholder="drjohn" {...field} className="bg-background border-input" />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="email"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Email <span className="text-destructive">*</span></FormLabel>
                  <FormControl>
                    <Input type="email" placeholder="john.doe@example.com" {...field} className="bg-background border-input"/>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="password"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Password {!isEditing && <span className="text-destructive">*</span>}</FormLabel>
                  <FormControl>
                    <Input type="password" placeholder={defaultValues ? "Leave blank to keep current" : "Enter password"} {...field} className="bg-background border-input"/>
                  </FormControl>
                  <FormDescription>
                    {defaultValues ? "Leave blank to keep the current password. New password must be at least 6 characters." : "Password must be at least 6 characters."}
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <div className="grid grid-cols-2 gap-4">
            <FormField
              control={form.control}
              name="phone"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Phone</FormLabel>
                  <FormControl>
                    <Input placeholder="555-123-4567" {...field} value={field.value ?? ''} className="bg-background border-input"/>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="department"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Department</FormLabel>
                  <FormControl>
                    <Input placeholder="Cardiology" {...field} value={field.value ?? ''} className="bg-background border-input"/>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            </div>
            <FormField
              control={form.control}
              name="role_id"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Role</FormLabel>
                  <Select 
                    onValueChange={(value) => field.onChange(value === NO_ROLE_VALUE ? null : parseInt(value))} 
                    value={field.value === null || field.value === undefined ? NO_ROLE_VALUE : field.value.toString()}
                    defaultValue={field.value === null || field.value === undefined ? NO_ROLE_VALUE : field.value.toString()}
                  >
                    <FormControl>
                      <SelectTrigger className="bg-background border-input">
                        <SelectValue placeholder="Select role" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value={NO_ROLE_VALUE}><em>None</em></SelectItem>
                      <SelectItem value={String(permissionToRoleId['Admin'])}>Admin</SelectItem>
                      <SelectItem value={String(permissionToRoleId['Doctor'])}>Doctor</SelectItem>
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
                    <Textarea placeholder="Optional notes about the doctor..." {...field} value={field.value ?? ''} className="bg-background border-input"/>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => { form.reset(); onOpenChange(false);}}>Cancel</Button>
              <Button type="submit">{defaultValues ? 'Save Changes' : 'Add Doctor'}</Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}
