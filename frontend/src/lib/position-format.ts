/** Display formatting for position episodes. Values stay full precision elsewhere. */

const MINUS = "−";

export function formatMoney(value: number | null | undefined, signed = false): string {
  if (value == null || !Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  const digits = abs >= 1000 ? 0 : 2;
  const body = abs.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  const prefix = value < 0 ? MINUS : signed && value > 0 ? "+" : "";
  return `${prefix}$${body}`;
}

/** Unit counts and prices: enough digits to tell fills apart, without scientific notation. */
export function formatNumber(value: number | null | undefined, signed = false): string {
  if (value == null || !Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  const digits = abs >= 1000 ? 0 : abs >= 10 ? 2 : abs === 0 ? 0 : 4;
  const body = abs.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  const prefix = value < 0 ? MINUS : signed && value > 0 ? "+" : "";
  return `${prefix}${body}`;
}

/** A fraction as a percent, dropping a needless ".0": 0.25 -> "25%", 0.3333 -> "33.3%". */
export function formatExposure(fraction: number | null | undefined): string {
  if (fraction == null || !Number.isFinite(fraction)) return "-";
  const text = (fraction * 100).toFixed(1).replace(/\.0$/, "");
  return `${text === "-0" ? "0" : text}%`;
}

export function formatSignedPercent(fraction: number | null | undefined): string {
  if (fraction == null || !Number.isFinite(fraction)) return "-";
  const pct = fraction * 100;
  const prefix = pct < 0 ? MINUS : pct > 0 ? "+" : "";
  return `${prefix}${Math.abs(pct).toFixed(2)}%`;
}
