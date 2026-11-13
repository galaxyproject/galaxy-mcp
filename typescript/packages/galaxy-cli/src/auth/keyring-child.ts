/**
 * One native keyring operation, run in its own short-lived process so the CLI can kill it.
 *
 * Why a process: `new AsyncEntry()` connects to the D-Bus session bus synchronously, and on a
 * bus that accepts the connection and never answers it blocks its thread forever
 * (https://github.com/Brooooooklyn/keyring-node/issues/156). A timer in the same process
 * cannot fire, and a `worker_thread` cannot be torn down either: the process cannot exit while
 * the native thread is blocked. A child process can be SIGKILLed whatever it is doing. gh gets
 * by with a goroutine and a timeout, because Go can abandon a call that never returns; a
 * blocked native thread in Node cannot be abandoned.
 *
 * Protocol: stdin is one JSON request, stdout is one JSON reply, and the process exits.
 *   request: { op: "get" | "set" | "delete", service, account, secret? }
 *   reply:   { ok: true, value?: string | null | boolean }
 *          | { ok: false, kind: "unavailable" | "not-found" | "failed", message }
 * `unavailable` is a keyring that cannot be reached (the module will not load, or the entry
 * cannot be built); `failed` is one that was reached and refused the call. A missing secret is
 * `ok` with `value: null`; the parent also maps `not-found` to that, but this child never sends it.
 * The secret travels only over these pipes, never argv, env or a file, and no reply or error
 * message echoes it. Standalone on purpose: it imports nothing from the CLI, so it builds as a
 * second tsup entry and also runs as source under Node's type stripping.
 */
import type * as Keyring from "@napi-rs/keyring";

export {};

interface KeyringRequest {
  op: "get" | "set" | "delete";
  service: string;
  account: string;
  secret?: string;
}

type Reply =
  | { ok: true; value: string | boolean | null }
  | { ok: false; kind: "unavailable" | "not-found" | "failed"; message: string };

async function run(req: KeyringRequest): Promise<Reply> {
  let AsyncEntry: typeof Keyring.AsyncEntry;
  // Imported here rather than at the top: the prebuilt binary is an optional dependency, and a
  // platform without one has to come back as "unavailable", not as a crashed helper.
  try {
    ({ AsyncEntry } = await import("@napi-rs/keyring"));
  } catch (e) {
    return {
      ok: false,
      kind: "unavailable",
      message: `the native keyring module could not be loaded: ${describe(e)}`,
    };
  }
  let entry: InstanceType<typeof AsyncEntry>;
  try {
    // Pinned to Secret Service. Left alone, the binding falls back to the kernel keyutils store
    // on any error, and a token kept there is gone at reboot; see keyring-store.ts.
    entry = new AsyncEntry(req.service, req.account, { linux: { store: "secret-service" } });
  } catch (e) {
    return { ok: false, kind: "unavailable", message: describe(e) };
  }
  try {
    // `?? null`: a missing entry comes back as null although the binding's types say undefined.
    if (req.op === "get") return { ok: true, value: (await entry.getPassword()) ?? null };
    if (req.op === "set") {
      if (typeof req.secret !== "string")
        return { ok: false, kind: "failed", message: "set without a secret" };
      await entry.setPassword(req.secret);
      return { ok: true, value: true };
    }
    return { ok: true, value: await entry.deletePassword() };
  } catch (e) {
    return { ok: false, kind: "failed", message: describe(e) };
  }
}

function describe(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const c of process.stdin) chunks.push(c as Buffer);
  return Buffer.concat(chunks).toString("utf8");
}

/** Whatever was on stdin is untrusted: `null`, a number or an object of the wrong shape all end up as "bad request". */
function isKeyringRequest(value: unknown): value is KeyringRequest {
  if (typeof value !== "object" || value === null) return false;
  const req = value as Partial<Record<keyof KeyringRequest, unknown>>;
  return (
    typeof req.op === "string" &&
    ["get", "set", "delete"].includes(req.op) &&
    typeof req.service === "string" &&
    typeof req.account === "string"
  );
}

let reply: Reply;
try {
  const req: unknown = JSON.parse(await readStdin());
  if (isKeyringRequest(req)) {
    reply = await run(req);
  } else {
    reply = { ok: false, kind: "failed", message: "bad request" };
  }
} catch {
  reply = { ok: false, kind: "failed", message: "bad request" };
}
// Exit from the write callback, not straight after the write: the Node docs warn that stdout
// writes can be asynchronous and that `process.exit()` skips whatever is still pending.
// eslint-disable-next-line n/no-process-exit -- the helper must end here, and only after the write callback
process.stdout.write(JSON.stringify(reply), () => process.exit(0));
