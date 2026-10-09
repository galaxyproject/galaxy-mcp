import type { GalaxyResult } from "@galaxyproject/galaxy-ops";

export type Format = "table" | "json" | "text";
export interface RenderOpts { format: Format; quiet: boolean; }

/**
 * How this surface PRINTS a result. Not how it measures one.
 *
 * The output budget is measured on the compact single line -- the same bytes the
 * MCP text block carries and the same bytes the Python server counts -- and only
 * then is the page printed indented. Measuring the indentation instead cut the
 * page a few rows shorter here than on the other two surfaces, which made the
 * budget a property of who was reading rather than of what a model reads. The
 * indented output can therefore run past 50,000 bytes, deliberately: whitespace
 * added for a human is not a reason to hand back a shorter page.
 */
export const printJson = (result: GalaxyResult<unknown>): string => JSON.stringify(result, null, 2);

export function render(result: GalaxyResult<unknown>, opts: RenderOpts): void {
  if (opts.format === "json") {
    console.log(printJson(result));
    return;
  }
  if (result.success) console.log(renderData(result.data));
  if (!opts.quiet && result.message) console.error(result.message);
  if (!opts.quiet && result.pagination?.helper_text) console.error(result.pagination.helper_text);
}

/**
 * The rows of a listing whose data is an object rather than a bare array.
 *
 * Most listings now put the page straight in `data`, which renders as a table
 * without any help. Three do not: get_tool_panel names its rows `entries` or
 * `tools` beside the section it opened, and get_history_contents names them
 * `contents` beside the history they came from. Keyed-value rendering would
 * print those as `entries [100]`, which is the shape of the answer rather than
 * the answer, so unwrap to the rows -- the window is on the message line.
 */
function pageRows(data: Record<string, unknown>): unknown[] | null {
  for (const key of ["entries", "tools", "contents"]) {
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
