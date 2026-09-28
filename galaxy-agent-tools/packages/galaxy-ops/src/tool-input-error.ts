/**
 * What the Python server says when Galaxy rejects a tool run over its inputs.
 *
 * A 400 from /api/tools is Galaxy refusing the tool form, and that server does not pass the
 * refusal on bare: it reads the tool's parameter list back, offers it, offers a structural
 * example from a tool test, and warns that Galaxy sometimes reports an input problem in
 * wording that sounds like a version incompatibility. All of that is one long sentence built
 * by `_format_tool_input_error` and `format_input_mismatch_error`, and it is the failure an
 * agent running a tool is most likely to meet.
 *
 * Two of its clauses are NOT here, and they are the two that need the input checker this
 * surface does not have: the list of inputs the schema proves wrong, and the keys the schema
 * does not model. Everything that does not depend on the check is ported -- which means the
 * sentence matches that server's whenever its checker found nothing to name, and is shorter
 * by those clauses when it did. Said plainly in the release notes rather than hidden here.
 *
 * server.py: `_is_credential_related_error`, `_format_run_tool_credential_error`,
 * `_format_tool_input_error`; tool_inputs.py: `format_input_mismatch_error`.
 */
import type { GalaxyContext } from "./context";
import { legacyGet } from "./legacy";
import { pyFormatError, pyLibraryText, type HttpFailureFacts } from "./python-failure";
import { schemaDescribesTool, schemaHasInputs } from "./tool-preflight";
import { summarizeToolInputs } from "./tool-inputs";

/** Where run_user_tool sends a caller left with nothing, since its tool is not in the toolbox. */
export const USER_TOOL_SHAPE_HINT =
  "Read this tool's representation from list_user_tools() for the expected shape " +
  "(a user-defined tool is not in the toolbox, so get_tool_input_template does not know it)";

/** Where a caller left with nothing is sent, for a tool the default calls cannot look up. */
export const SHAPE_HINT_DEFAULT =
  "Call get_tool_details(tool_id, io_details=True) (or get_tool_input_template(tool_id)) " +
  "to see the parameter schema";

/**
 * Whether a run failure appears to involve Galaxy credentials.
 *
 * A text scan, deliberately: it is what the other server does, over the same text -- the
 * client library's message with Galaxy's reply body in it -- because Galaxy has no status or
 * code that says "this was about credentials".
 */
export function isCredentialRelatedError(text: string): boolean {
  const lower = text.toLowerCase();
  return [
    "credential",
    "credentials",
    "credentials_context",
    "user_credentials",
    "service credential",
    "service credentials",
  ].some((marker) => lower.includes(marker));
}

/** The sentence for a run Galaxy refused while credentials were, or were not, in play. */
export function credentialFailureSentence(
  base: string,
  toolId: string,
  usedCredentials: boolean,
): string {
  if (usedCredentials) {
    return (
      `${base}. Galaxy rejected the run while using stored credentials for tool ` +
      `'${toolId}'. Check the configured credential values or active credential group ` +
      "for this tool, then try again."
    );
  }
  return (
    `${base}. This tool appears to require Galaxy tool credentials, but no stored ` +
    `credentials were found for tool '${toolId}'. Configure credentials for this tool ` +
    "in Galaxy, then retry the run."
  );
}

/** Python's `json.dumps(value, indent=2, default=str)`, for the two blocks that carry one. */
const pyJson = (value: unknown): string => JSON.stringify(value, null, 2);

/**
 * Assemble the enriched refusal.
 *
 * `schema` is for a caller that already holds the definition, as the user-tool run does: a
 * user tool is not in the toolbox, so every lookup by id would 404 and the message would come
 * back with nothing in it. Both reads are best-effort and neither can throw: a failure to
 * explain a failure still has to report the original one.
 */
export async function toolInputMismatchSentence(
  ctx: GalaxyContext,
  opts: {
    original: string;
    toolId: string;
    schema?: Record<string, unknown> | null;
    shapeHint?: string;
    toolVersion?: string | null;
  },
): Promise<string> {
  const heldSchema = opts.schema != null;
  let schema = opts.schema ?? null;
  let stale = false;
  if (!heldSchema) {
    try {
      schema = await legacyGet<Record<string, unknown>>(ctx, "/api/tools/{tool_id}", {
        params: {
          path: { tool_id: opts.toolId },
          query:
            opts.toolVersion == null
              ? { io_details: true, link_details: false }
              : { io_details: true, link_details: false, tool_version: opts.toolVersion },
        },
      });
    } catch {
      // Naming an input as wrong is as definitive as refusing to submit one, so the other
      // server re-reads the definition rather than trusting a cached copy, and when it
      // cannot it says nothing was confirmed. Nothing is cached here, so a failed read is
      // simply a read that did not happen.
      schema = null;
      stale = true;
    }
  }
  if (opts.toolVersion != null && schema !== null) {
    // The toolbox falls back to an installed version when the one asked for is missing, and
    // naming an input off another version's parameters is the same mistake as refusing on it.
    const served = schema["version"];
    if (typeof served === "string" && served !== opts.toolVersion) stale = true;
  }

  let summary: unknown[] | null = null;
  if (schema !== null && schemaHasInputs(schema)) {
    // An empty list dumped as "the expected parameters" is worse than saying nothing and
    // pointing at a call that can show them, so this only runs when there IS a list.
    summary = summarizeToolInputs(schema["inputs"] as unknown[]);
    // The check that would name which input looks wrong belongs here, gated on the schema
    // being the one that ran (`schemaDescribesTool`). It is not ported; see the header.
    void schemaDescribesTool;
  }

  let example: unknown = undefined;
  if (!heldSchema) {
    try {
      // Any version's example is fine -- it is a structural hint and nothing else.
      const tests = await legacyGet<unknown[]>(ctx, "/api/tools/{tool_id}/test_data", {
        params: { path: { tool_id: opts.toolId }, query: {} },
      });
      if (Array.isArray(tests) && tests.length > 0) {
        const first = tests[0];
        example =
          typeof first === "object" && first !== null
            ? (first as Record<string, unknown>)["inputs"]
            : undefined;
      }
    } catch {
      // Best-effort, exactly as over there.
    }
  }

  const lines: string[] = [
    opts.original,
    "",
    "This most likely means the `inputs` you provided do not match the parameter " +
      `schema for tool '${opts.toolId}'. It is not a sign that the MCP or the Galaxy ` +
      "version is incompatible. (Galaxy sometimes reports input problems as a " +
      'misleading "Required parameter(s) kwd not provided in request" error -- ' +
      "ignore that wording.)",
  ];
  if (summary !== null) {
    const caveat = stale ? " -- read from a copy that could not be refreshed" : "";
    lines.push(
      "",
      `Expected input parameters for '${opts.toolId}'${caveat} (build flattened keys ` +
        "like `section|param`, `cond|selector`, `repeat_0|param`):",
      pyJson(summary),
    );
  }
  if (example !== undefined) {
    lines.push(
      "",
      "Structural example from a tool test -- NOT runnable: the dataset IDs below " +
        "will not exist in your history. Copy the shape, not the values:",
      pyJson(example),
    );
  }
  if (summary === null && example === undefined) {
    lines.push("", `${opts.shapeHint ?? SHAPE_HINT_DEFAULT}, then rebuild \`inputs\` and retry.`);
  } else {
    lines.push("", "Rebuild `inputs` to match the schema above and call the tool again.");
  }
  return lines.join("\n");
}

/**
 * Turn a refused run into the sentence the other server would have said, or leave it.
 *
 * The three branches are its own, in its own order: a failure whose text mentions credentials
 * gets the credentials advice, a 400 gets the input-shape explanation, and anything else is
 * left for the op's failure contract to word. The message is replaced on the error itself,
 * which keeps its class -- and with it the kind a CLI exit code is read off -- and dropping
 * the request facts is what tells the boundary this sentence is finished.
 *
 * A 400 is the status Galaxy refuses a tool form with, which is why it decides this and not a
 * search for words in the reply; the other server reads it off the same exception field.
 */
export async function enrichedRunFailure(
  ctx: GalaxyContext,
  err: unknown,
  opts: {
    action: string;
    toolId: string;
    historyId: string;
    inputs: unknown;
    usedCredentials: boolean;
    schema?: Record<string, unknown> | null;
    shapeHint?: string;
    toolVersion?: string | null;
  },
): Promise<unknown> {
  const failure = err as { message?: string; http?: HttpFailureFacts };
  const facts = failure?.http;
  if (!facts) return err;
  const text = pyLibraryText("bioblend-write", facts);

  if (isCredentialRelatedError(text)) {
    const base = pyFormatError("Run tool", text, facts.status, {
      history_id: opts.historyId,
      tool_id: opts.toolId,
    });
    return finished(err, credentialFailureSentence(base, opts.toolId, opts.usedCredentials));
  }

  if (facts.status !== 400) return err;
  const original = pyFormatError(opts.action, text, facts.status, {
    history_id: opts.historyId,
    tool_id: opts.toolId,
    inputs: opts.inputs,
  });
  return finished(
    err,
    await toolInputMismatchSentence(ctx, {
      original,
      toolId: opts.toolId,
      schema: opts.schema,
      shapeHint: opts.shapeHint,
      toolVersion: opts.toolVersion,
    }),
  );
}

/** The same error, carrying a sentence that is already whole. */
function finished(err: unknown, sentence: string): unknown {
  const failure = err as { message: string; http?: HttpFailureFacts };
  failure.message = sentence;
  delete failure.http;
  return err;
}
