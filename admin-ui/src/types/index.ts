

export interface Doctor {
  id: string; 
  username: string; 
  password?: string; 
  email: string;
  phone: string | null;
  department: string | null;
  role_id: number | null; 
  notes?: string | null;
  create_time?: string;
  update_time?: string;
  patientCount?: number; 
}

export interface Patient {
  id: string; 
  username: string; 
  age: number | null;
  gender: 'Male' | 'Female' | 'Other' | null;
  case_id: string; 
  doctor_id: string | null;
  notes?: string | null; 
  create_time?: string;
  update_time?: string;
  attendingDoctorName?: string; 
  videoCount?: number;
  analysisCount?: number;
}

export interface DataAnalysisDataPoint {
  date: string; 
  analyses: number;
}

export interface DashboardMetrics {
  doctorCount: number;
  patientCount: number;
  videoCount: number;
  dataAnalysisCount: number;
}

export interface User { 
  id: string; 
  name: string; 
  email: string;
  avatarUrl?: string;
  role_id?: number; 
}

export interface Video {
  id: string; // video_id from VideoPath table
  video_path: string;
  create_time: string; // Upload date
  update_time: string;
  patient_id: string;
  patient_username?: string; // Joined from Patients table
  original_video: boolean;
  inference_video: boolean;
  action_id: string | null; // Link to Action table
  // thumbnail_url will be constructed on client: /api/v1/videos/thumbnail_image/{type}/{patient_id}/{video_id}
}

export interface Analysis { // Represents an Action from backend
  id: string; // action_id
  parent_id: string | null;
  video_id: string; // Original video_id from VideoPath table
  original_video_path?: string; // Joined from VideoPath table
  patient_id: string;
  patient_username?: string; // Joined from Patients table
  status: string;
  progress: string;
  create_time: string;
  update_time: string;
}

