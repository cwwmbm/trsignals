import { useSyncExternalStore } from "react";

export type RegimeConditionRequest = {
  id: number;
  left: string;
  operator: "is true" | "is false";
  label: string;
};

let current: RegimeConditionRequest | null = null;
let nextId = 1;
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((listener) => listener());
}

export function requestRegimeCondition(
  left: string,
  operator: RegimeConditionRequest["operator"],
  label: string,
) {
  current = { id: nextId, left, operator, label };
  nextId += 1;
  emit();
}

export function consumeRegimeCondition(id: number) {
  if (current?.id !== id) return;
  current = null;
  emit();
}

export function getRegimeConditionRequest() {
  return current;
}

export function subscribeRegimeConditionRequests(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useRegimeConditionRequest() {
  return useSyncExternalStore(subscribeRegimeConditionRequests, getRegimeConditionRequest);
}
