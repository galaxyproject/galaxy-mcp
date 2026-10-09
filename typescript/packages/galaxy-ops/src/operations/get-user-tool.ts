import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { GalaxyNotFoundError, httpError } from "../errors";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** One user-defined tool, as GET /api/unprivileged_tools/{uuid} answers it. */
export type UserToolRecord = GetJson<"/api/unprivileged_tools/{uuid}">;

// The shape Galaxy reports a user tool's uuid in: 32 hex digits in 8-4-4-4-12 groups. Galaxy
// binds the path value to a UUID column, so anything else is a 500 from the database driver
// rather than a 404 -- the shape is checked here so a caller reads "not found" either way.
// The same expression as server.py's _USER_TOOL_UUID, so both surfaces refuse the same strings.
const USER_TOOL_UUID = /^[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}$/;

const input = {
  uuid: z.string().describe("The tool's UUID as list_user_tools reports it: 32 hex digits in 8-4-4-4-12 groups."),
};
type In = { uuid: string };

async function run(i: In, ctx: GalaxyContext): Promise<UserToolRecord> {
  if (!USER_TOOL_UUID.test(i.uuid)) {
    // A refusal of our own, worded whole: no request went out, so there are no facts for the
    // failure contract to word a sentence from. server.py raises the same text as a ValueError.
    throw new GalaxyNotFoundError(
      `No user-defined tool found with UUID '${i.uuid}': that is not a UUID. A user tool's ` +
        "UUID is 32 hex digits in 8-4-4-4-12 groups, as list_user_tools() reports it. " +
        "Nothing was sent to Galaxy.",
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
