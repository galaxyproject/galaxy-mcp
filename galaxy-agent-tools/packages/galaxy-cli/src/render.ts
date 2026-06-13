import type { GalaxyResult } from "@galaxyproject/galaxy-ops";

export type Format = "table" | "json" | "text";
export interface RenderOpts { format: Format; quiet: boolean; }

export function render(result: GalaxyResult<unknown>, opts: RenderOpts): void {
  if (opts.format === "json") {
    console.log(JSON.stringify(result, null, 2));
    return;
  }
  if (result.success) console.log(renderData(result.data));
  if (!opts.quiet && result.message) console.error(result.message);
}

function renderData(data: unknown): string {
  if (data == null) return "";
  if (Array.isArray(data)) return data.length ? table(data as Record<string, unknown>[]) : "(empty)";
  if (typeof data === "object") return keyValue(data as Record<string, unknown>);
  return String(data);
}

function cell(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "object") return Array.isArray(v) ? `[${v.length}]` : "{...}";
  return String(v);
}

function table(rows: Record<string, unknown>[]): string {
  const cols = Array.from(new Set(rows.flatMap((r) => Object.keys(r)))).slice(0, 6);
  const widths = cols.map((c) => Math.max(c.length, ...rows.map((r) => cell(r[c]).length)));
  const line = (vals: string[]) => vals.map((v, i) => v.padEnd(widths[i] ?? 0)).join("  ");
  return [line(cols), line(cols.map((_, i) => "-".repeat(widths[i] ?? 0))), ...rows.map((r) => line(cols.map((c) => cell(r[c]))))].join("\n");
}

function keyValue(obj: Record<string, unknown>): string {
  const keys = Object.keys(obj).slice(0, 30);
  const w = Math.max(...keys.map((k) => k.length));
  return keys.map((k) => `${k.padEnd(w)}  ${cell(obj[k])}`).join("\n");
}
