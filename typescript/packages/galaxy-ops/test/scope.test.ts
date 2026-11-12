import { describe, it, expect } from "vitest";
import { createGalaxyContext, DEFAULT_POLL, type GalaxyContext } from "../src/context";
import { SCOPE_ENV, scopeFromEnv } from "../src/scope-config";
import {
  createAllowed,
  historyDeletable,
  scopeSummary,
  DEFAULT_HANDLE_CAPACITY,
  HANDLE_KINDS,
  HandleRegistry,
  allowedHistories,
  handleKindsOf,
  historyAllowed,
  historyIdOf,
  historyOutOfScopeSentence,
  kindMismatchSentence,
  recordHandles,
  recordRefs,
  recordUploadResult,
  wireName,
} from "../src/scope";

const withHandles = (handles = new HandleRegistry()): Pick<GalaxyContext, "handles"> => ({ handles });

describe("HandleRegistry", () => {
  it("is a Map from the lowercased id to the kinds it was returned as", () => {
    const ctx = withHandles();
    recordHandles(ctx, "job", ["J1", "j2"]);
    recordHandles(ctx, "hda", ["j1"]);
    expect(ctx.handles).toBeInstanceOf(Map);
    expect(ctx.handles!.get("j1")).toEqual(new Set(["job", "hda"]));
    expect(ctx.handles!.get("j2")).toEqual(new Set(["job"]));
    expect(handleKindsOf(ctx, "J1")).toEqual(new Set(["job", "hda"]));
    expect(handleKindsOf(ctx, "never")).toBeUndefined();
  });

  it("skips anything that is not a non-empty string", () => {
    const ctx = withHandles();
    recordHandles(ctx, "hda", [undefined, null, 7, "", { id: "x" }, "d1"]);
    expect([...ctx.handles!.keys()]).toEqual(["d1"]);
  });

  it("is bounded: past its capacity the least recently seen id is forgotten", () => {
    const ctx = withHandles(new HandleRegistry(3));
    recordHandles(ctx, "hda", ["a", "b", "c"]);
    recordHandles(ctx, "job", ["a"]); // seen again: now the most recent
    recordHandles(ctx, "hda", ["d"]); // over capacity: b, the oldest, goes
    expect([...ctx.handles!.keys()]).toEqual(["c", "a", "d"]);
    expect(ctx.handles!.size).toBe(3);
    expect(handleKindsOf(ctx, "b")).toBeUndefined();
    expect(handleKindsOf(ctx, "a")).toEqual(new Set(["hda", "job"]));
  });

  it("defaults to a capacity a long run stays under, and refuses a capacity below one", () => {
    expect(new HandleRegistry().capacity).toBe(DEFAULT_HANDLE_CAPACITY);
    expect(DEFAULT_HANDLE_CAPACITY).toBeGreaterThanOrEqual(1000);
    expect(() => new HandleRegistry(0)).toThrow(RangeError);
  });

  it("works on a plain Map a host supplies, with the default capacity", () => {
    const handles = new Map<string, Set<"job">>() as Map<string, Set<(typeof HANDLE_KINDS)[number]>>;
    const ctx = { handles };
    recordHandles(ctx, "job", ["j1"]);
    expect(handles.get("j1")).toEqual(new Set(["job"]));
  });

  it("records nothing on a context with no handle memory", () => {
    const ctx: Pick<GalaxyContext, "handles"> = {};
    recordHandles(ctx, "job", ["j1"]);
    recordRefs(ctx, [{ src: "hda", id: "d1" }]);
    recordUploadResult(ctx, { jobs: [{ id: "j1" }] });
    expect(ctx.handles).toBeUndefined();
  });

  it("reads {src, id} references and tool submissions", () => {
    const ctx = withHandles();
    recordRefs(ctx, [
      { src: "hda", id: "d1" },
      { src: "hdca", id: "c1" },
      { src: "dce", id: "e1" },
      { src: "job", id: "j1" },
      { src: "ldda", id: "l1" }, // a library dataset is not a kind the run remembers
      "not a ref",
      null,
    ]);
    recordUploadResult(ctx, {
      jobs: [{ id: "j2" }],
      outputs: [{ id: "d2" }, null],
      output_collections: [{ id: "c2" }],
      implicit_collections: [{ id: "c3" }],
    });
    expect(Object.fromEntries([...ctx.handles!].map(([k, v]) => [k, [...v]]))).toEqual({
      d1: ["hda"],
      c1: ["hdca"],
      e1: ["dataset_collection_element"],
      j1: ["job"],
      j2: ["job"],
      d2: ["hda"],
      c2: ["hdca"],
      c3: ["hdca"],
    });
  });
});

describe("the scope's decisions", () => {
  const scope = { historyIds: ["H1"], readOnlyHistoryIds: ["r1"] };

  it("lets a writable history be read and written, a read-only one only read, case-insensitively", () => {
    expect(historyAllowed(scope, "h1", "write")).toBe(true);
    expect(historyAllowed(scope, "h1", "read")).toBe(true);
    expect(historyAllowed(scope, "R1", "read")).toBe(true);
    expect(historyAllowed(scope, "R1", "write")).toBe(false);
    expect(historyAllowed(scope, "h2", "read")).toBe(false);
    expect(allowedHistories(scope, "write")).toEqual(["H1"]);
    expect(allowedHistories(scope, "read")).toEqual(["H1", "r1"]);
  });

  it("allows nothing under an empty scope", () => {
    expect(historyAllowed({ historyIds: [] }, "h1", "read")).toBe(false);
  });

  it("reads a record's history_id and nothing else", () => {
    expect(historyIdOf({ id: "d1", history_id: "h1" })).toBe("h1");
    expect(historyIdOf({ id: "d1" })).toBeUndefined();
    expect(historyIdOf({ history_id: "" })).toBeUndefined();
    expect(historyIdOf([{ history_id: "h1" }])).toBeUndefined();
    expect(historyIdOf(null)).toBeUndefined();
  });

  it("names parameters the way the wire spells them", () => {
    expect(wireName("historyId")).toBe("history_id");
    expect(wireName("datasetId")).toBe("dataset_id");
    expect(wireName("inputs")).toBe("inputs");
  });

  it("words a refusal with the parameter, the id and the allowed ids only", () => {
    expect(historyOutOfScopeSentence("get_history_details", "historyId", "h2", scope, "read")).toBe(
      "history_id 'h2' is outside this run's scope. Histories this run may read: H1, r1. " +
        "Nothing was sent to Galaxy.",
    );
    expect(historyOutOfScopeSentence("update_history", "historyId", "r1", scope, "write")).toBe(
      "history_id 'r1' is read-only in this run and update_history changes it. " +
        "Histories this run may write: H1; read-only: r1. Nothing was sent to Galaxy.",
    );
    expect(kindMismatchSentence("datasetId", "j1", new Set(["job"]), "hda")).toBe(
      "dataset_id 'j1' was returned by an earlier call as a job id; dataset_id expects a dataset id. " +
        "Nothing was sent to Galaxy.",
    );
    expect(kindMismatchSentence("collectionId", "x", new Set(["job", "hda"]), "hdca")).toContain(
      "as a job id and a dataset id;",
    );
  });
});

describe("createGalaxyContext with a scope", () => {
  const base = { baseUrl: "https://g.example", apiKey: "K" };

  it("has neither a scope nor a handle memory unless asked, as every context before it", () => {
    const ctx = createGalaxyContext(base);
    expect(ctx.scope).toBeUndefined();
    expect(ctx.handles).toBeUndefined();
    expect("scope" in ctx).toBe(false);
    expect("handles" in ctx).toBe(false);
    expect(ctx.poll).toEqual(DEFAULT_POLL);
  });

  it("copies and freezes the scope, so the host's later edits do not reach a running context", () => {
    const historyIds = ["h1"];
    const ctx = createGalaxyContext({ ...base, scope: { historyIds, blockedOperations: ["create_history"] } });
    historyIds.push("h2");
    expect(ctx.scope).toEqual({
      historyIds: ["h1"],
      readOnlyHistoryIds: [],
      blockedOperations: ["create_history"],
      includeCreated: true,
      allowCreate: true,
      destructive: "created",
    });
    expect(ctx.createdHistories).toEqual(new Set());
    expect(Object.isFrozen(ctx.scope)).toBe(true);
    expect(Object.isFrozen(ctx.scope!.historyIds)).toBe(true);
  });

  it("comes with a handle memory under a scope, takes the host's, or runs without one on null", () => {
    expect(createGalaxyContext({ ...base, scope: { historyIds: ["h1"] } }).handles).toBeInstanceOf(HandleRegistry);
    const shared = new HandleRegistry(50);
    expect(createGalaxyContext({ ...base, scope: { historyIds: ["h1"] }, handles: shared }).handles).toBe(shared);
    expect(createGalaxyContext({ ...base, handles: shared }).handles).toBe(shared); // the kind check alone
    expect(createGalaxyContext({ ...base, scope: { historyIds: ["h1"] }, handles: null }).handles).toBeUndefined();
  });

  it("refuses a scope whose ids are not non-empty strings", () => {
    expect(() => createGalaxyContext({ ...base, scope: { historyIds: [""] } })).toThrow(TypeError);
    expect(() => createGalaxyContext({ ...base, scope: { historyIds: [1 as never] } })).toThrow(TypeError);
    expect(() => createGalaxyContext({ ...base, scope: { historyIds: "h1" as never } })).toThrow(TypeError);
    expect(() => createGalaxyContext({ ...base, scope: { historyIds: [], destructive: "all" as never } })).toThrow(TypeError);
  });
});

describe("created histories and the destructive rule", () => {
  const scope = { historyIds: ["a1"], readOnlyHistoryIds: ["b2"] };

  it("counts a created history as readable and writable, and as the only deletable kind by default", () => {
    expect(historyAllowed(scope, "d4", "write", ["d4"])).toBe(true);
    expect(historyAllowed(scope, "d4", "write")).toBe(false);
    expect(allowedHistories(scope, "read", ["d4"])).toEqual(["a1", "d4", "b2"]);
    expect(historyDeletable(scope, "a1", ["d4"])).toBe(false);
    expect(historyDeletable(scope, "D4", ["d4"])).toBe(true);
    expect(historyDeletable({ ...scope, destructive: "writable" }, "a1", [])).toBe(true);
    expect(historyDeletable({ ...scope, destructive: "writable" }, "b2", [])).toBe(false);
  });

  it("allows creating unless switched off or create_history is blocked", () => {
    expect(createAllowed(scope)).toBe(true);
    expect(createAllowed({ ...scope, allowCreate: false })).toBe(false);
    expect(createAllowed({ ...scope, blockedOperations: ["create_history"] })).toBe(false);
  });

  it("summarises the scope for the session, created histories included", () => {
    const summary = scopeSummary({ scope, createdHistories: new Set(["d4"]) })!;
    expect(summary).toEqual({
      readable_history_ids: ["a1", "d4", "b2"],
      writable_history_ids: ["a1", "d4"],
      created_history_ids: ["d4"],
      create_allowed: true,
      destructive: "created",
      blocked_operations: [],
      note:
        "This session is scoped. It may read histories: a1, d4, b2; write to: a1, d4. " +
        "It may create histories, and use the ones it creates. It may delete only histories it created itself.",
    });
    expect(scopeSummary({})).toBeUndefined();
    expect(scopeSummary({ scope: { ...scope, includeCreated: false }, createdHistories: new Set(["d4"]) })!.writable_history_ids).toEqual(["a1"]);
  });
});

describe("scopeFromEnv", () => {
  it("is no scope at all when nothing is set", () => {
    expect(scopeFromEnv({})).toBeUndefined();
    expect(scopeFromEnv({ [SCOPE_ENV.write]: "  " })).toBeUndefined();
  });

  it("reads the histories, the creation switch, the destructive rule and the blocked tools", () => {
    expect(
      scopeFromEnv({
        [SCOPE_ENV.write]: "a1, b2",
        [SCOPE_ENV.read]: "c3",
        [SCOPE_ENV.allowCreate]: "0",
        [SCOPE_ENV.destructive]: "writable",
        [SCOPE_ENV.blocked]: "upload_file",
      }, ["upload_file"]),
    ).toEqual({
      historyIds: ["a1", "b2"],
      readOnlyHistoryIds: ["c3"],
      blockedOperations: ["upload_file"],
      includeCreated: true,
      allowCreate: false,
      destructive: "writable",
    });
    expect(scopeFromEnv({ [SCOPE_ENV.createdOnly]: "1" })).toMatchObject({ historyIds: [], allowCreate: true, destructive: "created" });
  });

  it("fails by name on anything it cannot read, rather than falling back to unscoped", () => {
    expect(() => scopeFromEnv({ [SCOPE_ENV.write]: "not-an-id" })).toThrow(/GALAXY_MCP_SCOPE_WRITE_HISTORIES.*'not-an-id'/);
    expect(() => scopeFromEnv({ [SCOPE_ENV.write]: "a1", [SCOPE_ENV.destructive]: "all" })).toThrow(/GALAXY_MCP_SCOPE_DESTRUCTIVE/);
    expect(() => scopeFromEnv({ [SCOPE_ENV.write]: "a1", [SCOPE_ENV.allowCreate]: "maybe" })).toThrow(/GALAXY_MCP_SCOPE_ALLOW_CREATE must be 1 or 0/);
    expect(() => scopeFromEnv({ [SCOPE_ENV.write]: "a1", [SCOPE_ENV.createdOnly]: "1" })).toThrow(/CREATED_ONLY=1/);
    expect(() => scopeFromEnv({ [SCOPE_ENV.createdOnly]: "1", [SCOPE_ENV.allowCreate]: "0" })).toThrow(/leaves no history/);
    expect(() => scopeFromEnv({ [SCOPE_ENV.destructive]: "writable" })).toThrow(/a scope needs histories/);
    expect(() => scopeFromEnv({ [SCOPE_ENV.write]: "a1", [SCOPE_ENV.blocked]: "nope" }, ["run_tool"])).toThrow(/unknown tools: nope/);
  });
});
