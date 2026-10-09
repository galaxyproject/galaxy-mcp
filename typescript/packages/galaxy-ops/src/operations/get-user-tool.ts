import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { GalaxyNotFoundError, httpError } from "../errors";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** One user-defined tool, as GET /api/unprivileged_tools/{uuid} answers it. */
export type UserToolRecord = GetJson<"/api/unprivileged_tools/{uuid}">;

const USER_TOOL_UUID_HEX = /^[0-9a-fA-F]{32}$/;

/**
 * Whether Galaxy's own uuid check (Python's `uuid.UUID(value)`) would accept `value`.
 *
 * Galaxy 26.0+ validates the path value with `uuid.UUID` and answers a 400 "Invalid UUID
 * format" for anything it refuses (older Galaxies bound it straight to the UUID column and
 * answered a 500 from the database driver). Neither is a 404 -- a 400 would classify as a
 * connection error here -- so the check is repeated and a malformed value reads as not found.
 * The spellings `uuid.UUID` accepts all compare on the same 32 hex digits, so the hyphenated
 * form list_user_tools reports, the bare 32-digit form, the braced form and `urn:uuid:` all
 * find the same tool. The same normalization as server.py's _is_user_tool_uuid, so both
 * surfaces refuse the same strings.
 */
function isUserToolUuid(value: string): boolean {
  const hex = value
    .replaceAll("urn:", "")
    .replaceAll("uuid:", "")
    .replace(/^[{}]+|[{}]+$/g, "")
    .replaceAll("-", "");
  return USER_TOOL_UUID_HEX.test(hex);
}

const input = {
  uuid: z
    .string()
    .describe(
      "The tool's UUID as list_user_tools reports it (32 hex digits in 8-4-4-4-12 groups). The bare 32-digit, braced and urn:uuid: spellings name the same tool and are accepted too.",
    ),
};
type In = { uuid: string };

async function run(i: In, ctx: GalaxyContext): Promise<UserToolRecord> {
  if (!isUserToolUuid(i.uuid)) {
    // A refusal of our own, worded whole: no request went out, so there are no facts for the
    // failure contract to word a sentence from. server.py raises the same text as a ValueError.
    throw new GalaxyNotFoundError(
      `No user-defined tool found with UUID '${i.uuid}': that is not a UUID. A user tool's ` +
        "UUID is 32 hex digits, as list_user_tools() reports it (hyphenated, bare, braced " +
        "or urn:uuid: spellings all name the same tool). Nothing was sent to Galaxy.",
    );
  }
  const { data, error, response } = await ctx.client.GET("/api/unprivileged_tools/{uuid}", {
    params: { path: { uuid: i.uuid } },
  });
  if (error || !data) throw httpError(response, error);
  return data;
}

export const getUserToolOp: Operation<typeof input, UserToolRecord> = {
  name: "get_user_tool",
  domain: "userTools",
  summary: "Get one user-defined tool by its uuid, with its full representation.",
  input,
  run,
  // server.py, get_user_tool: the tool id Galaxy answered with, or the uuid when the record
  // carries none (`or`, so an empty string falls back the same way on both sides).
  project: (out, i) => ({
    message: `Retrieved user-defined tool '${out.tool_id || i.uuid}' (UUID: ${i.uuid})`,
  }),
  // server.py, get_user_tool: a raw GET whose status it checks itself, so requests' own
  // text is what the sentence quotes.
  failure: {
    shape: "raise-for-status",
    action: "Get user tool",
    context: (i) => ({ uuid: i.uuid }),
  },
};

register(getUserToolOp as AnyOperation);

export const getUserTool = (i: In, ctx: GalaxyContext) => runOperation(getUserToolOp, i, ctx);
