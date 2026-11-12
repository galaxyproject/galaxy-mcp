/**
 * The scope guard's classification of every operation, pinned.
 *
 * Each op declares which of its parameters carry a Galaxy id and of what kind, and what a
 * scoped run does with it beyond that. This table is the whole policy in one place, so a new
 * op, or a changed one, moves it here deliberately -- and the registry's own lint, that a
 * parameter spelled like a known id parameter cannot go undeclared, is pinned beside it.
 */
import { describe, it, expect } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { z } from "zod";
import "../../src/operations/all";
import { createGalaxyContext, DEFAULT_POLL, type GalaxyContext } from "../../src/context";
import { KNOWN_ID_PARAMS, allOperations, register, runWithEnvelope } from "../../src/operations/registry";
import type { AnyOperation } from "../../src/operations/types";
import { getServerInfoOp } from "../../src/operations/get-server-info";
import { getUserOp } from "../../src/operations/get-user";
import { mockClient } from "../util/mock-client";

/** "ids | access | checks", the way the README tabulates it. */
function classification(op: AnyOperation): string {
  const ids = Object.entries(op.ids ?? {}).map(([param, kind]) => `${param}=${kind}`).join(", ") || "-";
  const access = op.scope?.access ?? (op.readOnly === false ? "write" : "read");
  const checks =
    [
      op.scope?.requires?.length
        ? `requires ${op.scope.requires.map((r) => (typeof r === "string" ? r : r.join("|"))).join(",")}`
        : "",
      op.scope?.creates ? "creates" : "",
      op.scope?.destroys ? "destroys" : "",
      op.scope?.annotate ? "annotate" : "",
      op.scope?.ownerOf ? "ownerOf" : "",
      op.scope?.owner ? "owner" : "",
      op.scope?.filter ? "filter" : "",
      op.scope?.restrict ? "restrict" : "",
    ]
      .filter(Boolean)
      .join(", ") || "-";
  return `${ids} | ${access} | ${checks}`;
}

const EXPECTED: Record<string, string> = {
  cancel_workflow_invocation: "invocationId=invocation | write | ownerOf, owner",
  create_history: "- | write | creates",
  create_page: "historyId=history | write | -",
  create_user_tool: "- | write | -",
  delete_user_tool: "- | write | -",
  download_dataset: "datasetId=hda | read | ownerOf",
  get_collection_details: "collectionId=hdca | read | owner",
  get_dataset_details: "datasetId=hda | read | owner",
  get_histories: "- | read | -",
  get_history_contents: "historyId=history | read | -",
  get_history_details: "historyId=history | read | -",
  get_invocations: "invocationId=invocation, workflowId=workflow, historyId=history | read | owner, filter",
  get_iwc_workflow_details: "- | read | -",
  get_iwc_workflows: "- | read | -",
  get_job_details: "datasetId=hda, historyId=history, jobId=job | read | owner",
  get_job_logs: "jobId=job | read | ownerOf",
  get_page: "- | read | -",
  get_page_revision: "- | read | -",
  get_server_info: "- | read | annotate",
  get_tool_citations: "- | read | -",
  get_tool_details: "- | read | -",
  get_tool_input_template: "- | read | -",
  get_tool_panel: "- | read | -",
  get_tool_run_examples: "- | read | -",
  get_user: "- | read | restrict",
  get_user_tool: "- | read | -",
  get_workflow_details: "workflowId=workflow | read | -",
  get_workflow_input_template: "workflowId=workflow, historyId=history | read | -",
  import_workflow_from_iwc: "- | write | -",
  invoke_workflow: "workflowId=workflow, historyId=history | write | requires historyId|historyName, creates",
  list_history_ids: "- | read | -",
  list_jobs: "historyId=history | read | filter",
  list_page_revisions: "- | read | -",
  list_pages: "historyId=history | read | -",
  list_user_tools: "- | read | -",
  list_workflows: "- | read | -",
  recommend_biocontainer: "- | read | -",
  recommend_iwc_workflows: "- | read | -",
  revert_page_revision: "- | write | -",
  run_tool: "historyId=history | write | -",
  run_user_tool: "historyId=history | write | -",
  search_iwc_workflows: "- | read | -",
  search_tools_by_keywords: "- | read | -",
  search_tools_by_name: "- | read | -",
  update_history: "historyId=history | write | destroys",
  update_page: "- | write | -",
  upload_file: "historyId=history | write | requires historyId",
  upload_file_from_url: "historyId=history | write | requires historyId",
};

describe("every operation's scope classification", () => {
  it("is the pinned table, op for op", () => {
    const actual = Object.fromEntries(
      allOperations.map((op) => [op.name, classification(op)]).sort(([a], [b]) => (a < b ? -1 : 1)),
    );
    expect(actual).toEqual(EXPECTED);
    expect(Object.keys(actual).length).toBe(48);
  });

  it("declares every parameter spelled like a known id parameter", () => {
    for (const op of allOperations) {
      for (const param of Object.keys(op.input)) {
        if (param in KNOWN_ID_PARAMS) {
          expect(op.ids?.[param], `${op.name}.${param}`).toBe(KNOWN_ID_PARAMS[param]);
        }
      }
    }
  });

  it("is frozen with the op, so a declaration cannot be edited after registration", () => {
    for (const op of allOperations) {
      if (op.ids) expect(Object.isFrozen(op.ids), op.name).toBe(true);
      if (op.scope) expect(Object.isFrozen(op.scope), op.name).toBe(true);
    }
  });
});

describe("the registry's lint on scope declarations", () => {
  const bare = (name: string, extra: Partial<AnyOperation>): AnyOperation =>
    ({
      name,
      domain: "histories",
      summary: "a test op",
      input: { historyId: z.string(), other: z.string() },
      run: async () => ({}),
      ...extra,
    }) as AnyOperation;

  it("refuses an op that takes a known id parameter without classifying it", () => {
    expect(() => register(bare("lint_undeclared", {}))).toThrow(
      /takes "historyId" but declares no `ids` entry for it/,
    );
  });

  it("refuses a declaration for a parameter the op does not take, or of a kind it does not know", () => {
    expect(() =>
      register(bare("lint_unknown_param", { ids: { historyId: "history", nope: "hda" } as never })),
    ).toThrow(/declares an id parameter "nope" its input does not have/);
    expect(() =>
      register(bare("lint_unknown_kind", { ids: { historyId: "banana" } as never })),
    ).toThrow(/unknown id kind "banana"/);
    expect(() =>
      register(bare("lint_unknown_required", { ids: { historyId: "history" }, scope: { requires: ["nope"] } })),
    ).toThrow(/requires a parameter "nope" its input does not have/);
  });

  it("registers an op that classifies what it takes", () => {
    const op = register(bare("lint_ok", { ids: { historyId: "history" } }));
    expect(op.name).toBe("lint_ok");
    expect(allOperations).toContain(op);
  });
});

describe("the API key stays out of every result", () => {
  const SRC = join(__dirname, "..", "..", "src");
  const files = (dir: string): string[] =>
    readdirSync(dir).flatMap((entry) => {
      const path = join(dir, entry);
      return statSync(path).isDirectory() ? files(path) : path.endsWith(".ts") ? [path] : [];
    });

  it("is read by the client, the upload path and the context, and by no result builder", () => {
    const mentioning = files(SRC)
      .filter((path) => /apiKey|api_key|x-api-key/.test(readFileSync(path, "utf8")))
      .map((path) => relative(SRC, path))
      .sort();
    // The client puts it on the wire, tus-upload and upload-file hand it to the resumable
    // upload, the context carries it, and legacy.ts names the header in a comment. No op's
    // run, project, failure contract or scope declaration reads it.
    expect(mentioning).toEqual(["client.ts", "context.ts", "legacy.ts", "operations/upload-file.ts", "tus-upload.ts"]);
  });

  it("appears in no envelope: not on a success, a scoped account read, or a failure", async () => {
    const KEY = "K-SECRET-VALUE-9f8e7d";
    const record = { id: "u1", username: "alice", email: "a@b.c", api_key: "another" };
    const answer = (body: unknown, status = 200) =>
      (async () =>
        new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } })) as unknown as typeof fetch;
    const scoped = createGalaxyContext({ baseUrl: "https://g.example", apiKey: KEY, fetchImpl: answer(record), scope: { historyIds: ["h1"] } });
    const open = createGalaxyContext({ baseUrl: "https://g.example", apiKey: KEY, fetchImpl: answer(record) });
    const failing = createGalaxyContext({ baseUrl: "https://g.example", apiKey: KEY, fetchImpl: answer({ err_msg: "no" }, 401) });
    for (const [ctx, op] of [
      [scoped, getUserOp],
      [open, getUserOp],
      [failing, getUserOp],
      [open, getServerInfoOp],
    ] as const) {
      const result = await runWithEnvelope(op as never, {} as never, ctx);
      expect(JSON.stringify(result)).not.toContain(KEY);
    }
    // And a context assembled by hand, whose client carries the key, hands none of it to an op.
    const ctx: GalaxyContext = { client: mockClient({ GET: () => ({ data: record, response: { status: 200 } }) }), poll: DEFAULT_POLL, apiKey: KEY };
    expect(JSON.stringify(await runWithEnvelope(getUserOp as never, {} as never, ctx))).not.toContain(KEY);
  });
});

/**
 * contract/scope.json is generated from the Python server's table
 * (`uv run python -m tests.scope_manifest` in python/). Everything this package declares --
 * the classification, the settings, the kinds, the sentences -- has to be what that file
 * says, so the two servers cannot scope a tool, or word a refusal, differently.
 */
describe("the scope contract shared with the Python server", async () => {
  const scopeModule = await import("../../src/scope");
  const { SCOPE_ENV } = await import("../../src/scope-config");
  const contract = JSON.parse(
    readFileSync(join(__dirname, "..", "..", "..", "..", "..", "contract", "scope.json"), "utf8"),
  ) as {
    env: Record<string, string>;
    idParams: Record<string, string>;
    kinds: Record<string, string>;
    handleCapacity: number;
    tools: Record<string, Record<string, unknown> & { ids: Record<string, string>; access: string; only?: string }>;
    sentences: Record<string, string>;
  };
  const wire = scopeModule.wireName;
  /** Listings narrowed inside the op, before paging, rather than by a declared filter. */
  const IN_TOOL = "histories_in_tool";

  it("classifies every tool both servers have the same way", () => {
    const ours = Object.fromEntries(
      allOperations
        .filter((op) => !op.name.startsWith("lint_"))
        .map((op) => {
          const row: Record<string, unknown> = {
            ids: Object.fromEntries(Object.entries(op.ids ?? {}).map(([param, kind]) => [wire(param), kind])),
            access: op.scope?.access ?? (op.readOnly === false ? "write" : "read"),
          };
          if (op.scope?.requires?.length) {
            row.requires = op.scope.requires.map((r) => (typeof r === "string" ? [r] : [...r]).map(wire));
          }
          for (const key of ["creates", "destroys", "ownerOf", "owner", "filter", "restrict", "annotate"] as const) {
            if (op.scope?.[key]) row[key] = true;
          }
          return [op.name, row];
        }),
    );
    const theirs = Object.fromEntries(
      Object.entries(contract.tools)
        .filter(([, row]) => row.only !== "python")
        .map(([name, row]) => {
          const flat: Record<string, unknown> = { ids: row.ids, access: row.access };
          if (row.requires) flat.requires = row.requires;
          for (const key of ["creates", "destroys", "ownerOf", "owner", "filter", "restrict", "annotate"]) {
            if (key in row && row[key] !== IN_TOOL) flat[key] = true;
          }
          return [name, flat];
        }),
    );
    expect(ours).toEqual(theirs);
    // The order of `ids` is part of it: a refusal names the first non-history id.
    for (const [name, row] of Object.entries(theirs)) {
      expect(Object.keys((ours[name] as { ids: object }).ids), name).toEqual(Object.keys((row as { ids: object }).ids));
    }
  });

  it("handles the tools only one server has explicitly", () => {
    const only = Object.entries(contract.tools).filter(([, row]) => row.only);
    expect(only.map(([name, row]) => `${name}:${row.only}`)).toEqual(["connect:python"]);
    expect(allOperations.some((op) => op.name === "connect")).toBe(false);
  });

  it("reads the same settings, kinds, id parameters and memory bound", () => {
    expect(SCOPE_ENV).toEqual(contract.env);
    expect(scopeModule.HANDLE_NOUNS).toEqual(contract.kinds);
    expect(Object.fromEntries(Object.entries(KNOWN_ID_PARAMS).map(([k, v]) => [wire(k), v]))).toEqual(contract.idParams);
    expect(scopeModule.DEFAULT_HANDLE_CAPACITY).toBe(contract.handleCapacity);
  });

  it("words every refusal the same, byte for byte", () => {
    const s = scopeModule;
    const scope = { historyIds: ["a1"], readOnlyHistoryIds: ["b2"] };
    const writable = { ...scope, destructive: "writable" as const };
    const created = ["d4"];
    expect({
      blocked: s.blockedOperationSentence("create_history"),
      history_out_of_scope_read: s.historyOutOfScopeSentence("get_history_details", "historyId", "c3", scope, "read", created),
      history_out_of_scope_write: s.historyOutOfScopeSentence("run_tool", "historyId", "c3", scope, "write", created),
      history_read_only: s.historyOutOfScopeSentence("update_history", "historyId", "b2", scope, "write", created),
      missing_param: s.missingScopedParamSentence("upload_file", "historyId", scope, created),
      missing_param_either: s.missingScopedParamSentence("invoke_workflow", ["historyId", "historyName"], scope, []),
      create_not_allowed: s.createNotAllowedSentence("create_history"),
      not_deletable_created: s.notDeletableSentence("update_history", "historyId", "a1", scope, created),
      not_deletable_writable: s.notDeletableSentence("update_history", "historyId", "c3", writable, created),
      owner_after: s.ownerOutOfScopeSentence("get_dataset_details", "dataset", "dd", "c3", scope, "read", "after", created),
      owner_before: s.ownerOutOfScopeSentence("cancel_workflow_invocation", "invocation", "ee", "c3", scope, "write", "before", []),
      owner_read_only: s.ownerOutOfScopeSentence("cancel_workflow_invocation", "invocation", "ee", "b2", scope, "write", "before", []),
      kind_mismatch: s.kindMismatchSentence("datasetId", "ee", new Set(["job"]), "hda"),
      kind_mismatch_two: s.kindMismatchSentence("collectionId", "ee", new Set(["job", "hda"]), "hdca"),
      summary_note: s.scopeSummary({ scope, createdHistories: new Set(created) })!.note,
    }).toEqual(contract.sentences);
  });
});
