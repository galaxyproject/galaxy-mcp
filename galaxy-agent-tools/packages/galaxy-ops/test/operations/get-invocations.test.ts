import { describe, it, expect } from "vitest";
import { z } from "zod";
import {
  getInvocationsOp,
  getInvocations,
  type InvocationSummary,
} from "../../src/operations/get-invocations";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyAuthError, GalaxyNotFoundError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

/** What /api/invocations returns per entry in the default 'collection' view. */
const summary = (id: string) => ({
  id,
  model_class: "WorkflowInvocation",
  workflow_id: "wf1",
  history_id: "hist1",
  state: "scheduled",
  create_time: "2026-09-17T10:00:00.000000",
  update_time: "2026-09-17T10:05:00.000000",
  uuid: `0000-${id}`,
});

const page = (n: number, from = 0) => Array.from({ length: n }, (_, k) => summary(`inv${from + k + 1}`));

/**
 * These tests are about the request get_invocations makes; the jobs it reads for an outcome
 * are the subject of their own tests below, so their requests are answered and left out here.
 */
const withoutRollup = (handlers: any) => ({
  ...handlers,
  GET: (path: string, init: any) =>
    path.endsWith("/jobs_summary")
      ? { data: { states: {} }, response: { status: 200 } }
      : handlers.GET(path, init),
});

const asList = (result: unknown): InvocationSummary[] => result as InvocationSummary[];

describe("get_invocations", () => {
  it("has parity name and returns the invocation", async () => {
    expect(getInvocationsOp.name).toBe("get_invocations");
    const client = mockClient(withoutRollup({
      GET: (path, init) => {
        expect(path).toBe("/api/invocations/{invocation_id}");
        expect(init.params.path.invocation_id).toBe("inv1");
        return { data: { id: "inv1", state: "scheduled", steps: [] }, response: { status: 200 } };
      },
    }));
    const inv = await getInvocations({ invocationId: "inv1" }, ctxWith(client));
    expect((inv as any).id).toBe("inv1");
  });

  it("takes the id and leaves the filters alone when both are given", async () => {
    const seen: string[] = [];
    const client = mockClient(withoutRollup({
      GET: (path, init) => {
        seen.push(path);
        expect(init.params.path.invocation_id).toBe("inv1");
        return { data: { id: "inv1", state: "ok" }, response: { status: 200 } };
      },
    }));
    const inv = await getInvocations(
      { invocationId: "inv1", workflowId: "wf1", historyId: "hist1", limit: 2, view: "element" },
      ctxWith(client),
    );
    expect(seen).toEqual(["/api/invocations/{invocation_id}"]);
    expect(Array.isArray(inv)).toBe(false);
    expect((inv as any).id).toBe("inv1");
  });

  it("lists when the id is an empty string, the way a falsy id does on the other surface", async () => {
    let path = "";
    const client = mockClient(withoutRollup({
      GET: (p) => {
        path = p;
        return { data: page(1), response: { status: 200 } };
      },
    }));
    expect(asList(await getInvocations({ invocationId: "" }, ctxWith(client)))).toHaveLength(1);
    expect(path).toBe("/api/invocations");
  });

  it("asks for one invocation's step details only when told to", async () => {
    const seen: any[] = [];
    const client = mockClient(withoutRollup({
      GET: (path, init) => {
        seen.push({ path, params: init.params });
        return { data: { id: "inv1", steps: [] }, response: { status: 200 } };
      },
    }));
    await getInvocations({ invocationId: "inv1", stepDetails: true }, ctxWith(client));
    await getInvocations({ invocationId: "inv1" }, ctxWith(client));
    expect(seen[0]).toEqual({
      path: "/api/invocations/{invocation_id}",
      params: { path: { invocation_id: "inv1" }, query: { step_details: true } },
    });
    expect(seen[1]).toEqual({
      path: "/api/invocations/{invocation_id}",
      params: { path: { invocation_id: "inv1" } },
    });
  });

  it("throws GalaxyNotFoundError on 404", async () => {
    const client = mockClient(withoutRollup({ GET: () => ({ error: {}, response: { status: 404 } }) }));
    await expect(getInvocations({ invocationId: "x" }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyNotFoundError,
    );
  });

  it("lists every invocation when no id and no filters are given", async () => {
    let seen: any;
    const client = mockClient(withoutRollup({
      GET: (path, init) => {
        seen = { path, query: init.params.query };
        return { data: page(3), response: { status: 200 } };
      },
    }));
    const result = asList(await getInvocations({}, ctxWith(client)));
    expect(seen.path).toBe("/api/invocations");
    expect(seen.query).toEqual({
      workflow_id: null,
      history_id: null,
      limit: null,
      view: "collection",
      step_details: false,
      include_terminal: true,
    });
    expect(result.map((i: any) => i.id)).toEqual(["inv1", "inv2", "inv3"]);
  });

  it("passes the filters, the limit and the detail level through to the index", async () => {
    let query: any;
    const client = mockClient(withoutRollup({
      GET: (_path, init) => {
        query = init.params.query;
        return { data: page(1), response: { status: 200 } };
      },
    }));
    await getInvocations(
      { workflowId: "wf1", historyId: "hist1", view: "element", stepDetails: true, limit: 5 },
      ctxWith(client),
    );
    expect(query).toMatchObject({
      workflow_id: "wf1",
      history_id: "hist1",
      view: "element",
      step_details: true,
      limit: 5,
    });
  });

  it("hands back the page the index sent, without a bound of its own", async () => {
    const client = mockClient(withoutRollup({ GET: () => ({ data: page(9), response: { status: 200 } }) }));
    const result = asList(await getInvocations({ limit: 4 }, ctxWith(client)));
    expect(result).toHaveLength(9);
  });

  it("returns a short page whole", async () => {
    const client = mockClient(withoutRollup({ GET: () => ({ data: page(2), response: { status: 200 } }) }));
    expect(asList(await getInvocations({ limit: 5 }, ctxWith(client)))).toHaveLength(2);
  });

  it("advertises the defaults it applies, and declares no limit of its own", () => {
    const schema = z.object(getInvocationsOp.input);
    expect(schema.parse({})).toEqual({ view: "collection", stepDetails: false });
    expect(() => schema.parse({ limit: 2.5 })).toThrow();
  });

  it("wants a real number for limit, like every other listing", () => {
    const schema = z.object(getInvocationsOp.input);
    // Coercion would take all three of these: "" and true become numbers Python would refuse,
    // and a one-element array becomes its element.
    expect(() => schema.parse({ limit: "" })).toThrow();
    expect(() => schema.parse({ limit: "5" })).toThrow();
    expect(() => schema.parse({ limit: [1] })).toThrow();
    expect(() => schema.parse({ limit: true })).toThrow();
    expect(schema.parse({ limit: 5 }).limit).toBe(5);
    expect(schema.parse({ limit: null }).limit).toBeNull();
  });

  it("reads an explicit null as not supplied, the way the other surface reads None", async () => {
    let query: any;
    const client = mockClient(withoutRollup({
      GET: (_path, init) => {
        query = init.params.query;
        return { data: page(1), response: { status: 200 } };
      },
    }));
    const schema = z.object(getInvocationsOp.input);
    const parsed = schema.parse({ invocationId: null, workflowId: null, historyId: null, limit: null });
    // The coercion is the trap: Number(null) is 0, and Galaxy rejects limit=0.
    expect(parsed.limit).toBeNull();
    await getInvocations(parsed, ctxWith(client));
    expect(query).toEqual({
      workflow_id: null,
      history_id: null,
      limit: null,
      view: "collection",
      step_details: false,
      include_terminal: true,
    });
  });

  it("treats a blank filter as no filter, the way the other surface's truthiness check does", async () => {
    let query: any;
    const client = mockClient(withoutRollup({
      GET: (_path, init) => {
        query = init.params.query;
        return { data: page(1), response: { status: 200 } };
      },
    }));
    await getInvocations({ workflowId: "", historyId: "" }, ctxWith(client));
    // Sent as "", Galaxy's encoded-id validator rejects the listing outright.
    expect(query.workflow_id).toBeNull();
    expect(query.history_id).toBeNull();
  });

  it("still refuses null where the other surface refuses it", () => {
    const schema = z.object(getInvocationsOp.input);
    // view and step_details are plain defaulted parameters there, with no null branch.
    expect(() => schema.parse({ view: null })).toThrow();
    expect(() => schema.parse({ stepDetails: null })).toThrow();
  });

  it("does not promise an unlimited listing, because omitting limit does not buy one", () => {
    // Galaxy's index applies its own default when the parameter is absent, so a description
    // saying "no limit" would send an agent off with an incomplete inventory.
    const described = getInvocationsOp.input.limit.description ?? "";
    expect(described).not.toMatch(/no limit/i);
    expect(described).toMatch(/Galaxy's own index default/);
  });

  it("returns an empty listing as an empty list", async () => {
    const client = mockClient(withoutRollup({ GET: () => ({ data: [], response: { status: 200 } }) }));
    expect(asList(await getInvocations({ limit: 5 }, ctxWith(client)))).toEqual([]);
  });

  it("raises the index's own error rather than reporting it as an empty list", async () => {
    const client = mockClient(withoutRollup({
      GET: () => ({ data: { err_msg: "History is not accessible" }, response: { status: 200 } }),
    }));
    await expect(getInvocations({}, ctxWith(client))).rejects.toThrow(/History is not accessible/);
  });

  it("says so when the index answers with something that is not a list at all", async () => {
    const client = mockClient(withoutRollup({ GET: () => ({ data: { anything: 1 }, response: { status: 200 } }) }));
    await expect(getInvocations({}, ctxWith(client))).rejects.toThrow(/did not return a list/);
  });

  it("classifies a failure on the listing path, not just on the detail path", async () => {
    const forbidden = mockClient(withoutRollup({ GET: () => ({ error: {}, response: { status: 403 } }) }));
    await expect(getInvocations({}, ctxWith(forbidden))).rejects.toBeInstanceOf(GalaxyAuthError);

    const broken = mockClient(withoutRollup({ GET: () => ({ error: { err_msg: "boom" }, response: { status: 500 } }) }));
    await expect(getInvocations({ workflowId: "wf1" }, ctxWith(broken))).rejects.toThrow(/boom/);
  });

  it("counts the list and reports the state for a single invocation", () => {
    const listMeta = getInvocationsOp.project!(page(2) as any, {} as any);
    expect(listMeta.message).toBe("Retrieved 2 workflow invocations");
    expect(listMeta.count).toBe(2);
    // Galaxy windows this index itself and reports no total, so there is no
    // window to describe -- the Python tool sends no pagination here either,
    // with or without a limit.
    expect(listMeta.pagination).toBeUndefined();

    // One row and the plural stays, because the other server writes it either way.
    const oneMeta = getInvocationsOp.project!(page(1) as any, { limit: 5 } as any);
    expect(oneMeta.message).toBe("Retrieved 1 workflow invocations");
    expect(oneMeta.count).toBe(1);
    expect(oneMeta.pagination).toBeUndefined();

    const detailMeta = getInvocationsOp.project!(
      { id: "inv1", state: "ok" } as any,
      { invocationId: "inv1" } as any,
    );
    expect(detailMeta.message).toBe("Retrieved invocation 'inv1'");
    expect(detailMeta.count).toBeUndefined();
    expect(detailMeta.pagination).toBeUndefined();
  });
});

describe("get_invocations outcome", () => {
  /** A Galaxy whose invocations have the jobs `states` says, or no readable summary. */
  function galaxy(invocation: Record<string, unknown> | Record<string, unknown>[], states?: unknown) {
    const asked: string[] = [];
    const client = mockClient({
      GET: (path: string) => {
        asked.push(path);
        if (path === "/api/invocations/{invocation_id}/jobs_summary") {
          if (states === undefined) throw new Error("500");
          return { data: { states }, response: { status: 200 } };
        }
        return { data: invocation, response: { status: 200 } };
      },
    });
    return { asked, ctx: { client, poll: DEFAULT_POLL } as never };
  }

  const outcomeOf = async (state: string, states?: unknown) =>
    ((await getInvocations({ invocationId: "i1" }, galaxy({ id: "i1", state }, states).ctx)) as any)
      .outcome;

  it("calls a completed run whose job failed failed", async () => {
    const g = galaxy({ id: "i1", state: "completed" }, { ok: 3, error: 1 });
    const out = (await getInvocations({ invocationId: "i1" }, g.ctx)) as Record<string, unknown>;
    expect(out).toMatchObject({ outcome: "failed", job_states: { ok: 3, error: 1 } });
  });

  it("calls a run failed once only the jobs Galaxy paused behind the failure remain", async () => {
    expect(await outcomeOf("scheduled", { ok: 2, error: 1, paused: 3 })).toBe("failed");
    expect(await outcomeOf("scheduled", { ok: 2, error: 1, paused: 2, running: 1 })).toBe("failing");
  });

  it("leaves Galaxy's state on paused jobs without a failure, and closes a run over a skipped one", async () => {
    expect(await outcomeOf("scheduled", { ok: 2, paused: 1 })).toBe("scheduled");
    expect(await outcomeOf("scheduled", { ok: 2, skipped: 1 })).toBe("completed");
  });

  it("gives no outcome when the summary cannot be read, rather than one guessed from no jobs", async () => {
    const out = (await getInvocations({ invocationId: "i1" }, galaxy({ id: "i1", state: "completed" }).ctx)) as any;
    expect(out).toEqual({ id: "i1", state: "completed" });
  });

  it("answers a listing as Galaxy sent it, without a request per row", async () => {
    const listed = Array.from({ length: 25 }, (_, n) => ({ id: `i${n}`, state: "scheduled" }));
    const g = galaxy(listed, { ok: 1 });
    const out = (await getInvocations({}, g.ctx)) as any[];
    expect(out).toEqual(listed);
    expect(g.asked).toEqual(["/api/invocations"]);
  });
});
