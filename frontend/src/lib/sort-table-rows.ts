export type SortDirection = "asc" | "desc";

function isEmpty(value: unknown): boolean {
  return value === null || value === undefined || value === "";
}

function comparePrimitive(a: unknown, b: unknown): number {
  if (typeof a === "number" && typeof b === "number") {
    return a - b;
  }
  if (typeof a === "boolean" && typeof b === "boolean") {
    return Number(a) - Number(b);
  }
  return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" });
}

export function compareRowValues(
  a: unknown,
  b: unknown,
  direction: SortDirection,
): number {
  const multiplier = direction === "asc" ? 1 : -1;
  const aEmpty = isEmpty(a);
  const bEmpty = isEmpty(b);
  if (aEmpty && bEmpty) return 0;
  if (aEmpty) return 1;
  if (bEmpty) return -1;
  if (typeof a === "object" || typeof b === "object") {
    return multiplier * String(a).localeCompare(String(b));
  }
  return multiplier * comparePrimitive(a, b);
}

export function sortTableRows<T extends Record<string, unknown>>(
  rows: T[],
  column: string,
  direction: SortDirection,
): T[] {
  return [...rows].sort((left, right) => compareRowValues(left[column], right[column], direction));
}

export function defaultSortDirectionForColumn(
  rows: Array<Record<string, unknown>>,
  column: string,
): SortDirection {
  if (column === "Sharpe" || column === "Sortino" || column === "CAGR" || column === "PnL") {
    return "desc";
  }
  const sample = rows.find((row) => !isEmpty(row[column]))?.[column];
  return typeof sample === "number" ? "desc" : "asc";
}

export function defaultSortColumn(
  columns: string[],
): string | undefined {
  if (columns.includes("Sharpe")) return "Sharpe";
  return columns[0];
}

/** Stable identity for row sets — avoids resetting sort when parent recreates the array. */
export function tableRowsSignature(rows: Array<Record<string, unknown>>): string {
  if (!rows.length) return "";
  const keys = Object.keys(rows[0]).sort();
  const fingerprint = (row: Record<string, unknown>) =>
    keys.map((key) => String(row[key] ?? "")).join("\u001f");
  return `${rows.length}\u001e${keys.join(",")}\u001e${fingerprint(rows[0])}\u001e${fingerprint(rows[rows.length - 1])}`;
}
