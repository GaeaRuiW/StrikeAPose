"use client";
import type { DataAnalysisDataPoint } from "@/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from "@/components/ui/chart";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, TooltipProps } from "recharts";
import { NameType, ValueType } from 'recharts/types/component/DefaultTooltipContent';

interface DataAnalysisChartProps {
  data: DataAnalysisDataPoint[];
}

const chartConfig = {
  analyses: {
    label: "Analyses",
    color: "hsl(var(--primary))",
  },
};

const CustomTooltip = ({ active, payload, label }: TooltipProps<ValueType, NameType>) => {
  if (active && payload && payload.length) {
    return (
      <div className="p-2 bg-background border border-border rounded-md shadow-lg">
        <p className="label text-sm font-medium">{`${label}`}</p>
        <p className="intro text-sm text-primary">{`Analyses : ${payload[0].value}`}</p>
      </div>
    );
  }
  return null;
};


export function DataAnalysisChart({ data }: DataAnalysisChartProps) {
  return (
    <Card className="shadow-lg rounded-lg overflow-hidden col-span-1 md:col-span-2 lg:col-span-4">
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-card-foreground">Data Analysis Trends</CardTitle>
        <CardDescription className="text-sm text-muted-foreground">Number of data analyses per day over the last few months.</CardDescription>
      </CardHeader>
      <CardContent className="h-[350px] w-full p-4">
        <ChartContainer config={chartConfig} className="h-full w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{
                top: 5,
                right: 20,
                left: 0, 
                bottom: 5,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
              <XAxis 
                dataKey="date" 
                tickLine={false} 
                axisLine={false} 
                tickMargin={8} 
                fontSize={12}
                stroke="hsl(var(--muted-foreground))"
              />
              <YAxis 
                tickLine={false} 
                axisLine={false} 
                tickMargin={8} 
                fontSize={12}
                stroke="hsl(var(--muted-foreground))"
              />
              <ChartTooltip
                cursor={{ stroke: "hsl(var(--accent))", strokeWidth: 1, strokeDasharray: "3 3" }}
                content={<ChartTooltipContent indicator="dot" hideLabel />}
              />
              <ChartLegend content={<ChartLegendContent />} />
              <Line
                type="monotone"
                dataKey="analyses"
                stroke="var(--color-analyses)"
                strokeWidth={2.5}
                dot={{
                  fill: "var(--color-analyses)",
                  r: 4,
                  strokeWidth: 2,
                  stroke: "hsl(var(--background))"
                }}
                activeDot={{
                  r: 6,
                  fill: "var(--color-analyses)",
                  stroke: "hsl(var(--background))",
                  strokeWidth: 2,
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
