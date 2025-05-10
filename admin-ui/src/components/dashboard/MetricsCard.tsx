import type { LucideIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

interface MetricsCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  description?: string;
}

export function MetricsCard({ title, value, icon: Icon, description }: MetricsCardProps) {
  return (
    <Card className="shadow-lg rounded-lg overflow-hidden hover:shadow-xl transition-shadow duration-300">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-semibold text-card-foreground">{title}</CardTitle>
        <Icon className="h-5 w-5 text-primary" />
      </CardHeader>
      <CardContent>
        <div className="text-3xl font-bold text-foreground">{value}</div>
        {description && <CardDescription className="text-xs text-muted-foreground pt-1">{description}</CardDescription>}
      </CardContent>
    </Card>
  );
}
