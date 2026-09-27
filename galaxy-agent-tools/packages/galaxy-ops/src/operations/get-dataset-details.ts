import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

export type DatasetDetail = GetJson<"/api/datasets/{dataset_id}">;

/**
 * Galaxy's own text peek, sliced to `preview_lines`.
 *
 * Three shapes, exactly as Python's tool returns them: the text case carries every key;
 * a datatype Galaxy has no text preview for carries `lines: null`, `error` and Galaxy's
 * flag; and a peek that could not be taken at all -- the fetch failed, what came back was
 * not the JSON object the route promises, or it could not be sliced into lines -- carries
 * only `lines: null` and `error`.
 */
export interface DatasetPreview {
  /** The sliced text, or null when there is none to show. */
  lines: string | null;
  /** How many lines that came to. */
  preview_lines?: number;
  /** Our line slice dropped some of the text Galaxy sent. */
  truncated?: boolean;
  /**
   * Galaxy's own `truncated` flag, verbatim. Galaxy sets it from the stored byte size while
   * reading up to 1,000,000 decoded characters, and those two disagree in both directions:
   * compressed text can be clipped with the flag false, and uncompressed multibyte text can
   * come back whole with the flag true. Read it as what Galaxy said, not as a statement
   * about how much of the dataset this is.
   */
  content_truncated_by_galaxy?: boolean;
  /** Why there are no lines. */
  error?: string;
}

/**
 * The dataset payload, with the preview alongside it when one was asked for and taken.
 *
 * Galaxy's OpenAPI types this route's 200 body as `unknown`, so `DatasetDetail` carries no
 * fields to intersect with and the record is spelled out here instead.
 */
export type DatasetDetailsResult = Record<string, unknown> & { preview?: DatasetPreview };

const DEFAULT_PREVIEW_LINES = 10;

const input = {
  datasetId: z.string().describe("Encoded dataset id"),
  includePreview: z
    .boolean()
    .default(true)
    .describe(
      "Whether to include a preview of the dataset content showing first N lines " +
        "(default: true, only works for datasets in 'ok' state)",
    ),
  previewLines: z
    .number()
    .int()
    .default(DEFAULT_PREVIEW_LINES)
    .describe(`Number of lines to include in the content preview (default: ${DEFAULT_PREVIEW_LINES})`),
};
type In = { datasetId: string; includePreview?: boolean; previewLines?: number };

/**
 * A JSON object, which is what Python's `body.get(...)` needs -- not a list, a string, a
 * number or null.
 */
function isJsonObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** What Galaxy sent instead, named for the preview error. */
function bodyKind(v: unknown): string {
  if (v === undefined) return "no body";
  if (v === null) return "null";
  if (Array.isArray(v)) return "an array";
  return `a ${typeof v}`;
}

/**
 * Galaxy's bounded text route, not the dataset.
 *
 * `get_content_as_text` reads at most about a megabyte from the start of the file
 * (`hda_manager.text_data`, MAX_PEEK_SIZE), so a ten-line preview of a 5 GB dataset costs
 * a megabyte rather than 5 GB. Python wraps the fetch, the parse AND the line slice in one
 * try/except and reports the failure in the preview rather than failing the call, and so
 * does this: nothing the peek can do costs the caller the metadata it came for.
 */
async function previewFor(
  ctx: GalaxyContext,
  datasetId: string,
  previewLines: number,
): Promise<DatasetPreview> {
  try {
    const { data, error, response } = await ctx.client.GET(
      "/api/datasets/{dataset_id}/get_content_as_text",
      { params: { path: { dataset_id: datasetId } } },
    );
    const httpOk = response.status >= 200 && response.status < 300;
    if (error || !httpOk) throw classifyHttp(response.status, error);
    // Python reads the reply with `body.get("item_data")`, so a 200 whose JSON is not an
    // object -- a list, a string, a number, null -- raises AttributeError there and comes back
    // as the preview error. Reading fields off one of those here instead would find them all
    // undefined and report a malformed reply as a datatype Galaxy cannot preview.
    if (!isJsonObject(data)) {
      throw new Error(`Galaxy sent ${bodyKind(data)} where the preview needs a JSON object`);
    }

    const cutByGalaxy = Boolean(data.truncated);
    // Read as unknown although the bindings type it `string | null`, because a 200 the
    // schema says cannot happen must not be the one thing that takes the whole call down.
    const itemData: unknown = data.item_data;
    if (itemData == null) {
      // text_data returns nothing for a datatype that is not text, and for a file that is
      // not where Galaxy expects it.
      return {
        lines: null,
        error: "No text preview: Galaxy previews text datatypes only",
        content_truncated_by_galaxy: cutByGalaxy,
      };
    }
    if (typeof itemData !== "string") {
      // Python reaches `content_str.split` and raises AttributeError here, which its own
      // except turns into the preview error below. Same answer, said out loud.
      throw new Error(`Galaxy sent item_data as ${typeof itemData}, not text`);
    }
    if (!Number.isInteger(previewLines)) {
      // Python slices with preview_lines as it stands, so a fractional or NaN one raises
      // TypeError from the slice and lands in the same except. Checked in the same place --
      // after the null case, which Python answers without ever slicing -- so both surfaces
      // report it the same way. A negative one is not an error on either side: it slices
      // from the end in Python and in JS alike.
      throw new Error(`preview_lines must be a whole number of lines, not ${previewLines}`);
    }

    // No total_lines: it only ever meant something while the whole file was downloaded, and
    // counting the lines of a peek reads as a count of the dataset.
    const all = itemData.split("\n");
    return {
      lines: all.slice(0, previewLines).join("\n"),
      preview_lines: Math.min(previewLines, all.length),
      truncated: all.length > previewLines,
      content_truncated_by_galaxy: cutByGalaxy,
    };
  } catch (e) {
    return { lines: null, error: `Preview unavailable: ${e instanceof Error ? e.message : String(e)}` };
  }
}

async function run(i: In, ctx: GalaxyContext): Promise<DatasetDetailsResult> {
  const { data, error, response } = await ctx.client.GET("/api/datasets/{dataset_id}", {
    params: { path: { dataset_id: i.datasetId } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  const dataset = data as DatasetDetail;

  const includePreview = i.includePreview ?? true;
  const previewLines = i.previewLines ?? DEFAULT_PREVIEW_LINES;
  const payload = dataset as Record<string, unknown>;
  if (!includePreview || payload.state !== "ok") return payload;

  return { ...payload, preview: await previewFor(ctx, i.datasetId, previewLines) };
}

export const getDatasetDetailsOp: Operation<typeof input, DatasetDetailsResult> = {
  name: "get_dataset_details",
  domain: "datasets",
  summary:
    "Show a dataset's metadata by id (state, extension, name), with an optional content preview. " +
    "The preview is Galaxy's own text peek -- up to about 1 MB of text read from the start of the " +
    "dataset, never the dataset itself -- sliced to previewLines lines, and only for a dataset in " +
    "the 'ok' state. preview.lines is null for a datatype Galaxy has no text preview for; " +
    "preview.truncated means the line slice cut the text Galaxy sent; and " +
    "preview.content_truncated_by_galaxy is Galaxy's own flag, set from the stored byte size " +
    "while the read counts decoded characters -- read it as what Galaxy said, not as a " +
    "statement about how much of the dataset this is.",
  input,
  run,
  project: (d) => ({ message: `Dataset ${(d as { id?: string }).id} state=${(d as { state?: string }).state}` }),
};

register(getDatasetDetailsOp as AnyOperation);

export const getDatasetDetails = (i: In, ctx: GalaxyContext) =>
  runOperation(getDatasetDetailsOp, i as InputOf<typeof input>, ctx);
