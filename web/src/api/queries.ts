import { useQuery } from "@tanstack/react-query";
import { api } from "./client";
import type {
  CompletionRecord,
  EvaluationResult,
  SessionTracePayload,
  TaskSummary,
  TopologyPayload,
  TrajectoryPayload,
} from "./types";

export function useTopology(intervalMs = 2000) {
  return useQuery<TopologyPayload>({
    queryKey: ["topology"],
    queryFn: () => api.get<TopologyPayload>("/api/topology"),
    refetchInterval: intervalMs,
  });
}

export function useTasks(intervalMs = 2000) {
  return useQuery<{ tasks: TaskSummary[] }>({
    queryKey: ["tasks"],
    queryFn: () => api.get<{ tasks: TaskSummary[] }>("/api/tasks?limit=200"),
    refetchInterval: intervalMs,
  });
}

export function useTask(taskId: string | undefined, intervalMs = 2000) {
  return useQuery({
    enabled: !!taskId,
    queryKey: ["task", taskId],
    queryFn: () => api.get<any>(`/api/tasks/${taskId}`),
    refetchInterval: intervalMs,
  });
}

export function useSession(sessionId: string | undefined, intervalMs = 2000) {
  return useQuery({
    enabled: !!sessionId,
    queryKey: ["session", sessionId],
    queryFn: () => api.get<any>(`/api/sessions/${sessionId}`),
    refetchInterval: intervalMs,
  });
}

export function useSessionTrajectory(sessionId: string | undefined) {
  return useQuery({
    enabled: !!sessionId,
    queryKey: ["session-trajectory", sessionId],
    queryFn: () => api.get<TrajectoryPayload>(`/api/sessions/${sessionId}/trajectory`),
  });
}

export function useSessionCompletions(sessionId: string | undefined, intervalMs = 4000) {
  return useQuery({
    enabled: !!sessionId,
    queryKey: ["session-completions", sessionId],
    queryFn: () =>
      api.get<{ session_id: string; completions: CompletionRecord[]; source: string }>(
        `/api/sessions/${sessionId}/completions`,
      ),
    refetchInterval: intervalMs,
  });
}

export function useSessionEvaluation(sessionId: string | undefined) {
  return useQuery({
    enabled: !!sessionId,
    queryKey: ["session-evaluation", sessionId],
    queryFn: () => api.get<EvaluationResult>(`/api/sessions/${sessionId}/evaluation`),
  });
}

export function useSessionRaw(sessionId: string | undefined) {
  return useQuery({
    enabled: !!sessionId,
    queryKey: ["session-raw", sessionId],
    queryFn: () => api.get<any>(`/api/sessions/${sessionId}/raw`),
  });
}

export function useSessionTrace(sessionId: string | undefined) {
  return useQuery({
    enabled: !!sessionId,
    queryKey: ["session-trace", sessionId],
    queryFn: () => api.get<SessionTracePayload>(`/api/sessions/${sessionId}/trace`),
    retry: false,
  });
}
