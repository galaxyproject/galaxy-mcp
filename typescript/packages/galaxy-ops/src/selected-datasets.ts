/**
 * Read a chosen set of a history's datasets by id, in the order they were chosen.
 *
 * Not an operation: nothing registers it as a tool, so the shared surface and its parity with
 * the Python server are unchanged. It is the read behind galaxy-mcp's optional
 * `show_datasets` card, where an agent picks the datasets and Galaxy supplies what they are.
 *
 * It asks the contents index for exactly these datasets with its `type_id-in` filter, so a
 * chosen id from another history simply does not come back. The legacy `?ids=` filter would
 * be the obvious read, but Galaxy (26.1) decodes those ids twice and rejects every one. An id
 * Galaxy cannot decode at all fails the read with Galaxy's own message.
 */
import type { GalaxyContext } from "./context";
import { GalaxyValidationError, httpError } from "./errors";

export const MAX_SELECTED_DATASETS = 20;
/** Long enough for any real dataset name; a longer one is cut so twenty always fit the budget. */
const MAX_NAME_LENGTH = 200;

export interface SelectedDatasetsArgs {
  historyId: string;
  /** Dataset ids as get_history_contents reports them (`id`, not `hid`). */
  datasetIds: string[];
}

export interface SelectedDatasetFields {
  id: string;
  hid?: number;
  name?: string;
  extension?: string;
  state?: string;
  deleted?: boolean;
  visible?: boolean;
  create_time?: string;
  update_time?: string;
  file_size?: number;
  /** Galaxy's own flag; absent when Galaxy did not send it. */
  accessible?: boolean;
  /** Set when `name` was cut to MAX_NAME_LENGTH characters. */
  name_truncated?: boolean;
}

/**
 * One chosen id and what became of it: `available` with its fields; `inaccessible` when Galaxy
 * returned it but says the user cannot access it; `unavailable` when it did not come back from
 * this history at all (not there, not a dataset, or deleted from the database).
 */
export interface SelectedDataset {
  id: string;
  status: "available" | "inaccessible" | "unavailable";
  dataset?: SelectedDatasetFields;
}

export interface SelectedDatasets {
  history_id: string;
  selected: SelectedDataset[];
  available: number;
}

const FIELDS = ["hid", "name", "extension", "state", "deleted", "visible", "create_time", "update_time", "file_size", "accessible"] as const;

function project(item: Record<string, unknown>, id: string): SelectedDatasetFields {
  const out: Record<string, unknown> = { id };
  for (const key of FIELDS) if (item[key] !== undefined && item[key] !== null) out[key] = item[key];
  if (typeof out.name === "string" && out.name.length > MAX_NAME_LENGTH) {
    out.name = out.name.slice(0, MAX_NAME_LENGTH);
    out.name_truncated = true;
  }
  return out as unknown as SelectedDatasetFields;
}

export async function getSelectedDatasets(args: SelectedDatasetsArgs, ctx: GalaxyContext): Promise<SelectedDatasets> {
  // A repeated id is shown once, where it was first chosen.
  const ids = [...new Set(args.datasetIds)];
  if (ids.length === 0) throw new GalaxyValidationError("dataset_ids must name at least one dataset");
  if (ids.length > MAX_SELECTED_DATASETS) {
    throw new GalaxyValidationError(`dataset_ids can name at most ${MAX_SELECTED_DATASETS} datasets (got ${ids.length})`);
  }

  const { data, error, response } = await ctx.client.GET("/api/histories/{history_id}/contents", {
    params: {
      path: { history_id: args.historyId },
      query: {
        v: "dev",
        q: ["type_id-in"],
        qv: [ids.map((id) => `dataset-${id}`).join(",")],
        keys: ["id", ...FIELDS].join(","),
      },
    },
  });
  if (error || !Array.isArray(data)) throw httpError(response, error);

  const returned = new Map<string, Record<string, unknown>>();
  for (const item of data as Record<string, unknown>[]) if (typeof item?.id === "string") returned.set(item.id, item);
  const selected = ids.map((id): SelectedDataset => {
    const item = returned.get(id);
    if (!item) return { id, status: "unavailable" };
    // Galaxy's inaccessible view still names the item, so it is shown; only an explicit
    // `accessible: true` counts as access.
    return { id, status: item.accessible === true ? "available" : "inaccessible", dataset: project(item, id) };
  });
  return { history_id: args.historyId, selected, available: selected.filter((s) => s.status === "available").length };
}
