import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyNotFoundError } from "../errors";
import { legacyGet } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

/** Hand-typed: Galaxy's classic tool API is not in the OpenAPI bindings (see legacy.ts). */
export interface ToolDetail {
  id: string;
  name: string;
  version?: string;
  description?: string;
  inputs?: unknown[];
  [extra: string]: unknown;
}

const input = {
  toolId: z.string().describe("Tool id, e.g. 'cat1' or 'toolshed.../fastqc/0.74'"),
  ioDetails: z.boolean().default(false).describe("Include full input/output details (default false)"),
  toolVersion: z
    .string()
    .nullish()
    .describe(
      "Describe this installed version of the tool rather than the one Galaxy picks for the id. " +
        "Refused when Galaxy answers with a different version, so the answer is never about another one.",
    ),
};
type In = { toolId: string; ioDetails?: boolean; toolVersion?: string | null };

async function run(i: In, ctx: GalaxyContext): Promise<ToolDetail> {
  // GET /api/tools/{tool_id} is the classic controller, which the 26.0 bindings do not
  // type, so tool_version rides the same untyped call as the rest of the query. Galaxy
  // reads it out of the query string (v26.1.1 api/tools.py show).
  const tool = await legacyGet<ToolDetail>(ctx, "/api/tools/{tool_id}", {
    params: {
      path: { tool_id: i.toolId },
      query: {
        io_details: i.ioDetails ?? false,
        ...(i.toolVersion != null ? { tool_version: i.toolVersion } : {}),
      },
    },
  });
  if (i.toolVersion != null) {
    // server.py, get_tool_details: the toolbox answers with its newest installed
    // version when the one asked for is missing, as a 200 with no word about it, so
    // the mismatch is refused after the request and the sentence is the tool's own.
    const served = tool.version;
    if (typeof served === "string" && served !== i.toolVersion) {
      throw new GalaxyNotFoundError(
        `Galaxy described version ${served} of tool '${i.toolId}' rather than the ` +
          `${i.toolVersion} asked for, so that version is not installed on this server. ` +
          `Call get_tool_details('${i.toolId}') without tool_version to see the version ` +
          "Galaxy serves for this id.",
      );
    }
  }
  return tool;
}

export const getToolDetailsOp: Operation<typeof input, ToolDetail> = {
  name: "get_tool_details",
  domain: "tools",
  summary: "Show a Galaxy tool's metadata by id (name, version, description). Legacy endpoint.",
  input,
  run,
  // server.py, get_tool_details: the id that was asked for. The name and the version
  // Galaxy answered with are in data, and the id here is the one to ask again with.
  // A pinned version is named too, now that the record is known to be that version's.
  project: (_t, i) => ({
    message:
      i.toolVersion != null
        ? `Retrieved details for tool '${i.toolId}' at version ${i.toolVersion}`
        : `Retrieved details for tool '${i.toolId}'`,
  }),
  // server.py, get_tool_details: the flag is in the context too, as Python's bool, and
  // the version only when one was asked for.
  failure: {
    shape: "bioblend-get",
    action: "Get tool details",
    context: (i) => ({
      tool_id: i.toolId,
      io_details: i.ioDetails ?? false,
      ...(i.toolVersion != null ? { tool_version: i.toolVersion } : {}),
    }),
  },
};

register(getToolDetailsOp as AnyOperation);

// A library caller may leave the defaulted arguments out; run() applies the same
// values the schema declares for the parsed surface path.
export const getToolDetails = (i: In, ctx: GalaxyContext) =>
  runOperation(getToolDetailsOp, i as InputOf<typeof input>, ctx);
