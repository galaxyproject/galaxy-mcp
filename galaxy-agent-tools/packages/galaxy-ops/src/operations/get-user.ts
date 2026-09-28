import type { GalaxyContext } from "../context";
import { httpError, GalaxyAuthError } from "../errors";
import { pyGet, pyStr } from "../python-values";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/**
 * The user record Galaxy sent, whole.
 *
 * The Python tool answers with what `/api/users/current` returned and picks no fields
 * out of it, so neither does this: a DetailedUserModel carries the disk usage, the
 * quota, the tags in use and the stored preferences, and a caller reading any of them
 * should not have to ask Galaxy a second time. The three named fields are the ones
 * anything reading a user needs and the ones the guard below insists on; everything
 * else rides along under the index signature.
 */
export type CurrentUser = Record<string, unknown> & {
  id: string;
  email: string;
  username: string;
};

const input = {}; // no args

async function run(_in: Record<string, never>, ctx: GalaxyContext): Promise<CurrentUser> {
  const { data, error, response } = await ctx.client.GET("/api/users/{user_id}", {
    params: { path: { user_id: "current" } },
  });
  if (error || !data) throw httpError(response, error);
  // The endpoint returns DetailedUserModel | AnonUserModel; the anonymous shape has
  // no id/email/username. Surface that as an auth error rather than silently
  // returning undefined fields.
  const u = data as { id?: string; email?: string; username?: string };
  if (!u.id || !u.email || !u.username) {
    throw new GalaxyAuthError("Anonymous user response -- a valid API key is required");
  }
  return data as CurrentUser;
}

export const getUserOp: Operation<typeof input, CurrentUser> = {
  name: "get_user", // parity: mcp-server-galaxy-py get_user
  domain: "connection",
  summary: "Return the current authenticated Galaxy user, as Galaxy's own user record.",
  input,
  run,
  // server.py, get_user: the username through dict.get, so a record without one is
  // announced as "unknown". run() has already refused an anonymous reply, so the
  // fallback is unreachable on this path and is transcribed rather than relied on.
  project: (u) => ({ message: `Retrieved user info for '${pyStr(pyGet(u, "username", "unknown"))}'` }),
  // server.py, get_user: its own sentence, with neither a hint nor a context.
  failure: { shape: "bioblend-get", sentence: (text) => `Failed to get user: ${text}` },
};

register(getUserOp as AnyOperation);

/** Code-mode entry. */
export const getUser = (i: Record<string, never>, ctx: GalaxyContext) => runOperation(getUserOp, i, ctx);
