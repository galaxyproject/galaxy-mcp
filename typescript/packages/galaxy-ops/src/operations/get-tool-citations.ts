import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyConnectionError, GalaxyNotFoundError } from "../errors";
import { legacyGet } from "../legacy";
import { pyFormatError } from "../python-failure";
import { pyStr } from "../python-values";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** Hand-typed: Galaxy's tool-show endpoint is not in the OpenAPI bindings. */
interface ToolShowPayload {
  id: string;
  name?: string;
  version?: string;
  [k: string]: unknown;
}

/**
 * One entry from `GET /api/tools/{id}/citations`, also off-schema. Galaxy answers bibtex
 * entries, plus a `format: "error"` entry for an id it can't resolve.
 */
export type ToolCitation =
  | { format: "bibtex"; content: string }
  | { format: "error"; error: string; tool_id: string };

export interface ToolCitationsResult {
  tool_name?: string;
  tool_version?: string;
  citations: ToolCitation[];
}

const input = {
  toolId: z.string().describe("tool id"),
};
type In = { toolId: string };

async function run(i: In, ctx: GalaxyContext): Promise<ToolCitationsResult> {
  // The show payload has no citations; it's read for name/version and so an unknown id
  // 404s -- the citations endpoint answers 200 with an error entry for one.
  const payload = await legacyGet<ToolShowPayload>(ctx, "/api/tools/{tool_id}", {
    params: {
      path: { tool_id: i.toolId },
      query: { io_details: false, link_details: false },
    },
  });
  // Galaxy takes no version here and resolves the default one, as the show did, so
  // tool_version is the version these citations belong to.
  const citations = await legacyGet<ToolCitation[]>(ctx, "/api/tools/{tool_id}/citations", {
    params: { path: { tool_id: i.toolId } },
  });
  // server.py, get_tool_citations: anything but a list is not a citations reply, and
  // measuring it would count its keys. Python raises a bare Exception for this and for the
  // error entry below, which carries no status field, so the hint falls to the text search.
  if (!Array.isArray(citations)) {
    throw new GalaxyConnectionError(
      pyFormatError(
        "Get tool citations",
        "the citations reply was not a list",
        null,
        { tool_id: i.toolId },
        { statusIsKnown: false },
      ),
    );
  }
  // Galaxy writes an error entry only when its toolbox can't resolve the id; a DOI that won't
  // resolve comes back as a placeholder bibtex instead. The show just resolved the same id the
  // same way, so this is a race -- a toolbox reload, or another worker that hasn't loaded the
  // tool -- and the whole answer is missing, not one citation.
  const refused = citations.find(
    (c): c is Extract<ToolCitation, { format: "error" }> => c?.format === "error",
  );
  if (refused) {
    // Galaxy's sentence ends in a full stop and the context clause starts with one.
    const said = pyStr(refused.error);
    throw new GalaxyNotFoundError(
      pyFormatError(
        "Get tool citations",
        said.endsWith(".") ? said.slice(0, -1) : said,
        null,
        { tool_id: i.toolId },
        { statusIsKnown: false },
      ),
    );
  }
  return {
    tool_name: payload.name,
    tool_version: payload.version,
    citations,
  };
}

export const getToolCitationsOp: Operation<typeof input, ToolCitationsResult> = {
  name: "get_tool_citations",
  domain: "tools",
  result: { kind: "object", fields: ["tool_name", "tool_version", "citations"] },
  summary: "Return citations for a Galaxy tool by id.",
  input,
  run,
  // server.py, get_tool_citations: the id that was asked for, not the name Galaxy
  // answered with, and no "(s)" -- the other server writes the plural either way.
  project: (out, i) => ({
    message: `Retrieved ${out.citations.length} citations for tool '${i.toolId}'`,
    count: out.citations.length,
  }),
  // server.py, get_tool_citations.
  failure: {
    shape: "bioblend-get",
    action: "Get tool citations",
    context: (i) => ({ tool_id: i.toolId }),
  },
};

register(getToolCitationsOp as AnyOperation);

export const getToolCitations = (i: In, ctx: GalaxyContext) => runOperation(getToolCitationsOp, i, ctx);
