/**
 * IWC (Intergalactic Workflow Commission) manifest utilities.
 *
 * Fetches and memoizes the public workflow manifest, exposes enrichment
 * helpers used by the IWC ops.  This is NOT an op -- it carries no
 * registration and is not exported from index.ts.
 */
import { GalaxyConnectionError } from "./errors";
import { pySplitWhitespace, pyStrip } from "./python-str";
import { stepsInOrder } from "./workflow-steps";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface IwcWorkflow {
  trsID: string;
  definition: {
    name?: string;
    annotation?: string;
    tags?: string[];
    steps?: Record<string, unknown> | unknown[];
    creator?: unknown;
    license?: string;
  };
  readme?: string;
  categories?: string[];
  updated?: string;
  [k: string]: unknown;
}

export interface IwcAuthor {
  name: string;
  orcid: string;
}

export interface EnrichedIwcWorkflow {
  trsID: string;
  name: string;
  description: string;
  tags: string[];
  readme_summary: string;
  readme?: string;
  step_count: number;
  authors: IwcAuthor[];
  categories: string[];
  license: string;
  tools_used: string[];
}

// ---------------------------------------------------------------------------
// Module-level memo (mirrors Python @lru_cache(maxsize=1))
// ---------------------------------------------------------------------------

let _cached: IwcWorkflow[] | null = null;

/** Reset the memo -- for tests only. */
export function __resetIwcCacheForTest(): void {
  _cached = null;
}

/** Prime the memo with a fixture -- for tests only. */
export function __setIwcCacheForTest(wfs: IwcWorkflow[]): void {
  _cached = wfs;
}

// ---------------------------------------------------------------------------
// Manifest fetch + flatten
// ---------------------------------------------------------------------------

const MANIFEST_URL = "https://iwc.galaxyproject.org/workflow_manifest.json";

/**
 * Fetch (and memoize for the process lifetime) the full flattened IWC
 * workflow list.  Mirrors Python `_fetch_iwc_workflows`.
 */
export async function fetchIwcWorkflows(): Promise<IwcWorkflow[]> {
  if (_cached !== null) return _cached;

  const resp = await globalThis.fetch(MANIFEST_URL);
  if (!resp.ok) {
    // The other server fetches this with requests and checks it with raise_for_status, so
    // every tool that reads the manifest quotes requests' own text for a refusal. The facts
    // travel with the error and each op's own sentence wraps them -- this fetch does not go
    // through the Galaxy client, so nothing else would have recorded them.
    const failure = new GalaxyConnectionError(
      `IWC manifest fetch failed: ${resp.status} ${resp.statusText}`,
      resp.status,
    );
    failure.http = {
      status: resp.status,
      method: "GET",
      url: MANIFEST_URL,
      bodyText: await resp.text(),
      reason: resp.statusText,
    };
    throw failure;
  }
  const manifest = (await resp.json()) as Array<{ workflows?: unknown[] }>;

  const all: IwcWorkflow[] = [];
  for (const entry of manifest) {
    if (Array.isArray(entry.workflows)) {
      for (const wf of entry.workflows) {
        all.push(wf as IwcWorkflow);
      }
    }
  }

  _cached = all;
  return all;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Extract deduplicated tool names from workflow steps, first use first.
 * Mirrors Python `_extract_tool_names_from_steps`.
 *
 * `steps` may be a Record<string, unknown> (Galaxy .ga format) or an array --
 * handle both. The order the steps are walked in is the order the names come out
 * in, so it is stated rather than inherited from the runtime's object: see
 * `stepsInOrder`.
 */
export function extractToolNamesFromSteps(steps: Record<string, unknown> | unknown[]): string[] {
  const values: unknown[] = Array.isArray(steps)
    ? steps
    : stepsInOrder(steps).map(([, step]) => step);
  const seen = new Set<string>();
  const result: string[] = [];

  for (const stepData of values) {
    if (!stepData || typeof stepData !== "object") continue;
    const step = stepData as Record<string, unknown>;
    const toolId = step["tool_id"];
    if (typeof toolId !== "string" || !toolId) continue;

    const parts = toolId.split("/");
    // toolshed format: take the second-to-last segment; else the whole id
    const toolName = parts.length > 1 ? parts[parts.length - 2] : toolId;
    if (toolName && !seen.has(toolName)) {
      seen.add(toolName);
      result.push(toolName);
    }
  }

  return result;
}

/**
 * Strip markdown headers and collapse whitespace into a ~300-character plain summary.
 *
 * `_clean_readme_summary`, line for line, and the lines that look like plain JavaScript are
 * the ones that were not:
 *
 *  - `trim`/`trimStart` is not `str.strip()`. A readme that opens with a byte order mark is
 *    the everyday case -- `trimStart` eats the U+FEFF and reads `# Title` as a heading to
 *    drop, while `strip` leaves it and the other server keeps the line, mark and all. The
 *    C0 information separators go the other way. `pyStrip` is that set.
 *  - `split(/\s+/)` is not `str.split()`: same disagreement about which characters separate,
 *    plus the empty pieces a regex split leaves at the ends.
 *  - `length` is UTF-16 code units and `len()` is code points, so a summary carrying one
 *    emoji is a character longer here than there and the cut lands in a different place --
 *    and a slice by code unit can cut a surrogate pair in half, which puts half a character
 *    on the wire. Both the measurement and the cut are by code point.
 */
export function cleanReadmeSummary(readme: string | undefined, maxLength = 300): string {
  if (!readme) return "";

  const lines = readme.split("\n");
  const cleanLines: string[] = [];

  for (const line of lines) {
    const stripped = pyStrip(line);
    if (stripped.startsWith("#")) continue;
    if (cleanLines.length === 0 && stripped === "") continue;
    cleanLines.push(line);
  }

  let text = pySplitWhitespace(cleanLines.join(" ")).join(" ");

  const chars = Array.from(text);
  if (chars.length > maxLength) {
    const truncated = chars.slice(0, maxLength - 3).join("");
    // `rsplit(" ", 1)[0]`: everything before the last space, or the whole of it when there
    // is no space to split on -- including the empty string when the only space is first.
    const lastSpace = truncated.lastIndexOf(" ");
    text = (lastSpace >= 0 ? truncated.slice(0, lastSpace) : truncated) + "...";
  }

  return text;
}

/**
 * Enrich a raw IWC manifest entry with computed metadata fields.
 * Mirrors Python `_enrich_workflow_result`.
 */
export function enrichWorkflowResult(wf: IwcWorkflow, opts: { fullReadme?: boolean } = {}): EnrichedIwcWorkflow {
  const definition = wf.definition ?? {};

  const steps = definition.steps;
  const stepsRecord: Record<string, unknown> | unknown[] | undefined = steps;

  // step_count: Python only counts dict steps (returns 0 for arrays)
  const stepCount =
    steps !== undefined && !Array.isArray(steps) && typeof steps === "object" ? Object.keys(steps).length : 0;

  // authors from definition.creator (may be an array or scalar)
  let authors: IwcAuthor[] = [];
  const creators = definition.creator;
  if (Array.isArray(creators)) {
    authors = creators
      .filter((c): c is Record<string, unknown> => c !== null && typeof c === "object")
      .map((c) => ({
        name: typeof c["name"] === "string" ? c["name"] : "",
        orcid: typeof c["identifier"] === "string" ? c["identifier"] : "",
      }));
  }

  const tools_used =
    stepsRecord !== undefined && !Array.isArray(stepsRecord) && typeof stepsRecord === "object"
      ? extractToolNamesFromSteps(stepsRecord as Record<string, unknown>)
      : [];

  const result: EnrichedIwcWorkflow = {
    trsID: wf.trsID ?? "",
    name: definition.name ?? "",
    description: definition.annotation ?? "",
    tags: Array.isArray(definition.tags) ? definition.tags : [],
    readme_summary: cleanReadmeSummary(wf.readme),
    step_count: stepCount,
    authors,
    categories: Array.isArray(wf.categories) ? wf.categories : [],
    license: definition.license ?? "",
    tools_used,
  };

  if (opts.fullReadme) {
    result.readme = wf.readme ?? "";
  }

  return result;
}
