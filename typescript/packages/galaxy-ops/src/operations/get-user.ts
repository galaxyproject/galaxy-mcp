import type { GalaxyContext } from "../context";
import { httpError } from "../errors";
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
 * anything reading a user needs; everything else rides along under the index signature.
 *
 * All three are OPTIONAL, because the endpoint answers `DetailedUserModel | AnonUserModel`
 * and the anonymous half has none of them -- the bindings give it three fields,
 * `nice_total_disk_usage`, `quota_percent` and `total_disk_usage`, and that is the whole
 * model. A session with no key gets that reply, the tool hands it back as it arrived (which
 * is what the other server does, naming the user "unknown" in its sentence), and a type
 * promising three strings would let `user.username.toLowerCase()` compile and then throw at
 * runtime. A caller that needs them has to check, which is the truth about this endpoint
 * rather than a concession to it.
 */
export type CurrentUser = Record<string, unknown> & {
  id?: string;
  email?: string;
  username?: string;
};

const input = {}; // no args

async function run(_in: Record<string, never>, ctx: GalaxyContext): Promise<CurrentUser> {
  const { data, error, response } = await ctx.client.GET("/api/users/{user_id}", {
    params: { path: { user_id: "current" } },
  });
  if (error || !data) throw httpError(response, error);
  // The endpoint returns DetailedUserModel | AnonUserModel, and the anonymous shape has no
  // id, email or username. This used to refuse that reply as an auth failure; the other
  // server answers it -- the record goes out as it arrived and the sentence names the user
  // as "unknown", which is what `dict.get("username", "unknown")` puts there. Python is the
  // contract, so this answers too: a caller reading `data.username` finds it missing, which
  // is the same thing the other server tells them, and one surface refusing what the other
  // returns is the difference that matters more.
  return data as CurrentUser;
}

export const getUserOp: Operation<typeof input, CurrentUser> = {
  name: "get_user", // parity: Python server get_user
  domain: "connection",
  summary: "Return the current authenticated Galaxy user, as Galaxy's own user record.",
  input,
  run,
  // server.py, get_user: the username through dict.get, so a record without one is
  // announced as "unknown" -- which is the reachable branch and not a transcribed one,
  // since the anonymous reply is answered rather than refused.
  project: (u) => ({ message: `Retrieved user info for '${pyStr(pyGet(u, "username", "unknown"))}'` }),
  // server.py, get_user: its own sentence, with neither a hint nor a context.
  failure: { shape: "bioblend-get", sentence: (text) => `Failed to get user: ${text}` },
};

register(getUserOp as AnyOperation);

/** Code-mode entry. */
export const getUser = (i: Record<string, never>, ctx: GalaxyContext) => runOperation(getUserOp, i, ctx);
