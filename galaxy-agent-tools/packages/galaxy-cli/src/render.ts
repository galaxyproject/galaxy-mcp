import type { GalaxyResult } from "@galaxyproject/galaxy-ops";

export type Format = "table" | "json" | "text";
export interface RenderOpts { format: Format; quiet: boolean; }

/**
 * How this surface serialises a result, and therefore what the output budget has
 * to be measured against: the indented JSON below is the largest thing render can
 * print, so a page that fits this fits the table and text formats too.
 */
export const serializeForCli = (result: GalaxyResult<unknown>): string => JSON.stringify(result, null, 2);

export function render(result: GalaxyResult<unknown>, opts: RenderOpts): void {
  if (opts.format === "json") {
    console.log(serializeForCli(result));
    return;
  }
  if (result.success) console.log(renderData(result.data));
  if (!opts.quiet && result.message) console.error(result.message);
  if (!opts.quiet && result.pagination?.helperText) console.error(result.pagination.helperText);
}

/**
 * The rows of a paged op's result, if that is what this is.
 *
 * A bounded list op returns its page under `items`, and get_tool_panel under
 * `entries` or `tools`, alongside a `pagination` object. Keyed-value rendering
 * would print that as `items [100]`, which is the shape of the result rather
 * than the result, so unwrap to the rows and let `pagination` go to the
 * message line instead of becoming a column.
 */
function pageRows(data: Record<string, unknown>): unknown[] | null {
  if (!("pagination" in data)) return null;
  for (const key of ["items", "entries", "tools"]) {
    const rows = data[key];
    if (Array.isArray(rows)) return rows;
  }
  return null;
}

function renderData(data: unknown): string {
  if (data == null) return "";
  if (Array.isArray(data)) return data.length ? table(data as Record<string, unknown>[]) : "(empty)";
  if (typeof data === "object") {
    const rows = pageRows(data as Record<string, unknown>);
    if (rows) return rows.length ? table(rows as Record<string, unknown>[]) : "(empty)";
    return keyValue(data as Record<string, unknown>);
  }
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
