import { describe, it, expect } from "vitest";
import { z } from "zod";
import { listJobsOp, listJobs } from "../../src/operations/list-jobs";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import {
  GalaxyAuthError,
  GalaxyConnectionError,
  GalaxyNotFoundError,
  GalaxyValidationError,
} from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

/** What /api/jobs returns per job in the default 'collection' view. */
const summary = (id: string, state = "ok") => ({
  id,
  model_class: "Job",
  state,
  tool_id: "cat1",
  exit_code: 0,
  create_time: "2026-01-01T00:00:00",
  update_time: "2026-01-01T00:05:00",
});

const page = (n: number, from = 0) => Array.from({ length: n }, (_, k) => summary(`job${from + k + 1}`));

describe("list_jobs", () => {
  it("has the parity name and is read-only", () => {
    expect(listJobsOp.name).toBe("list_jobs");
    expect(listJobsOp.domain).toBe("jobs");
    expect(listJobsOp.readOnly ?? true).toBe(true);
    expect(listJobsOp.requires).toBeUndefined();
  });

  it("sends the window and the defaults, and no filter it was not given", async () => {
    let seen: any;
    const client = mockClient({
      GET: (path, init) => {
        seen = { path, query: init.params.query };
        return { data: page(3), response: { status: 200 } };
      },
    });
    const jobs = await listJobs({}, ctxWith(client));
    expect(seen.path).toBe("/api/jobs");
    expect(seen.query).toEqual({
      limit: 100,
      offset: 0,
      history_id: null,
      state: null,
      date_range_min: null,
      date_range_max: null,
      order_by: "update_time",
      view: "collection",
    });
    expect(jobs.map((j) => j.id)).toEqual(["job1", "job2", "job3"]);
  });

  it("passes every filter, the sort and the view through to the index, in the Python order", async () => {
    let query: any;
    const client = mockClient({
      GET: (_path, init) => {
        query = init.params.query;
        return { data: page(1), response: { status: 200 } };
      },
    });
    await listJobs(
      {
        historyId: "hist1",
        state: "ok,error",
        dateRangeMin: "2026-01-01T00:00:00",
        dateRangeMax: "2026-02-01",
        orderBy: "create_time",
        view: "admin_job_list",
        limit: 5,
        offset: 10,
      },
      ctxWith(client),
    );
    expect(query).toEqual({
      limit: 5,
      offset: 10,
      history_id: "hist1",
      // One string, as it was given: Galaxy reads a comma-separated list, and the other
      // surface sends the same bytes rather than a repeated parameter.
      state: "ok,error",
      date_range_min: "2026-01-01T00:00:00",
      date_range_max: "2026-02-01",
      order_by: "create_time",
      view: "admin_job_list",
    });
    // The order the Python tool builds its params dict in, so the two surfaces ask the
    // same question in the same words.
    expect(Object.keys(query)).toEqual([
      "limit",
      "offset",
      "history_id",
      "state",
      "date_range_min",
      "date_range_max",
      "order_by",
      "view",
    ]);
  });

  it("treats a blank filter as no filter, the way the other surface's truthiness check does", async () => {
    let query: any;
    const client = mockClient({
      GET: (_path, init) => {
        query = init.params.query;
        return { data: [], response: { status: 200 } };
      },
    });
    await listJobs({ historyId: "", state: "", dateRangeMin: "", dateRangeMax: "" }, ctxWith(client));
    expect(query.history_id).toBeNull();
    expect(query.state).toBeNull();
    expect(query.date_range_min).toBeNull();
    expect(query.date_range_max).toBeNull();
  });

  it("refuses a bad window before asking Galaxy anything, and caps nothing", async () => {
    let asked = 0;
    const client = mockClient({
      GET: () => {
        asked += 1;
        return { data: [], response: { status: 200 } };
      },
    });
    await expect(listJobs({ limit: 0 }, ctxWith(client))).rejects.toThrow(
      new GalaxyValidationError("limit must be at least 1 (got 0)"),
    );
    await expect(listJobs({ offset: -1 }, ctxWith(client))).rejects.toThrow(
      new GalaxyValidationError("offset must be 0 or greater (got -1)"),
    );
    expect(asked).toBe(0);
    // Galaxy's job index has no ceiling, and neither does the Python tool.
    await listJobs({ limit: 5000 }, ctxWith(client));
    expect(asked).toBe(1);
  });

  it("hands back the page the index sent, without a bound of its own", async () => {
    const client = mockClient({ GET: () => ({ data: page(9), response: { status: 200 } }) });
    expect(await listJobs({ limit: 4 }, ctxWith(client))).toHaveLength(9);
  });

  it("returns an empty listing as an empty list", async () => {
    const client = mockClient({ GET: () => ({ data: [], response: { status: 200 } }) });
    expect(await listJobs({ historyId: "hist1" }, ctxWith(client))).toEqual([]);
  });

  it("reads an unknown history as an empty page, which is what Galaxy answers", async () => {
    // Galaxy filters the index by history_id without looking the history up: a well-formed
    // id nothing answers to is 200 [], the same bytes as a history with no jobs, never a 404.
    const client = mockClient({ GET: () => ({ data: [], response: { status: 200 } }) });
    expect(await listJobs({ historyId: "nothing" }, ctxWith(client))).toEqual([]);
    expect(listJobsOp.summary).toContain("get_history_details");
  });

  it("classifies the statuses the index does emit the way the exit codes need", async () => {
    // 403: a history the key cannot read (ItemAccessibilityException). 400: a filter Galaxy
    // could not read, a history_id that is not an encoded id among them (MalformedId).
    const forbidden = mockClient({ GET: () => ({ error: {}, response: { status: 403 } }) });
    await expect(listJobs({ historyId: "nope" }, ctxWith(forbidden))).rejects.toBeInstanceOf(GalaxyAuthError);
    const malformed = mockClient({ GET: () => ({ error: {}, response: { status: 400 } }) });
    await expect(listJobs({ historyId: "not-an-id" }, ctxWith(malformed))).rejects.toBeInstanceOf(
      GalaxyConnectionError,
    );
    // A 404 is not something this index says about a history, but the classifier still holds.
    const missing = mockClient({ GET: () => ({ error: {}, response: { status: 404 } }) });
    await expect(listJobs({}, ctxWith(missing))).rejects.toBeInstanceOf(GalaxyNotFoundError);
  });

  it("raises the index's own error rather than reporting it as an empty list", async () => {
    const client = mockClient({
      GET: () => ({ data: { err_msg: "History is not accessible by user", err_code: 403002 }, response: { status: 200 } }),
    });
    const failure = listJobs({ historyId: "hist1" }, ctxWith(client));
    await expect(failure).rejects.toBeInstanceOf(GalaxyConnectionError);
    await expect(failure).rejects.toThrow(
      "List jobs failed: History is not accessible by user. Context: err_code=403002",
    );
  });

  it("says so when the index answers with something that is not a list at all", async () => {
    const client = mockClient({ GET: () => ({ data: { anything: 1 }, response: { status: 200 } }) });
    await expect(listJobs({}, ctxWith(client))).rejects.toThrow(/did not return a list/);
  });

  it("counts the page and sends no pagination block", () => {
    const meta = listJobsOp.project!(page(2), {} as any);
    expect(meta.message).toBe("Retrieved 2 jobs");
    expect(meta.count).toBe(2);
    // Galaxy windows this index itself and reports no total, so there is no window to
    // describe -- the Python tool sends no pagination here either.
    expect(meta.pagination).toBeUndefined();
    // One row keeps the plural, because the other server writes it either way.
    expect(listJobsOp.project!(page(1), {} as any).message).toBe("Retrieved 1 jobs");
    expect(listJobsOp.project!([], {} as any)).toEqual({ message: "Retrieved 0 jobs", count: 0 });
  });

  it("words a failed request the way the Python tool does", () => {
    expect(listJobsOp.failure).toEqual({
      shape: "bioblend-get",
      action: "List jobs",
      context: expect.any(Function),
      sentence: expect.any(Function),
    });
    expect(listJobsOp.failure!.context!({ historyId: "hist1" } as any)).toEqual({ history_id: "hist1" });
  });

  it("follows a 400 with what it means, and leaves every other status to format_error", () => {
    const sentence = listJobsOp.failure!.sentence!;
    const text = "GET: error 400: b'...', 0 attempts left: ...";
    expect(sentence(text, 400, { historyId: "not-an-id" } as any)).toBe(
      `List jobs failed: ${text}. Context: history_id=not-an-id. Galaxy could not read one of the filters: ` +
        "history_id has to be an encoded id, the dates ISO 8601, and state, order_by and view values Galaxy " +
        "knows. A well-formed id of a history that does not exist is not refused this way -- it answers with " +
        "an empty page",
    );
    expect(sentence(text, 403, { historyId: "h" } as any)).toBeUndefined();
    expect(sentence(text, 500, {} as any)).toBeUndefined();
    expect(sentence(text, null, {} as any)).toBeUndefined();
  });

  it("advertises the defaults it applies", () => {
    const schema = z.object(listJobsOp.input);
    expect(schema.parse({})).toEqual({ orderBy: "update_time", view: "collection", limit: 100, offset: 0 });
  });

  it("wants a whole number for limit and offset, with no coercion", () => {
    const schema = z.object(listJobsOp.input);
    expect(() => schema.parse({ limit: 2.5 })).toThrow();
    expect(() => schema.parse({ limit: "5" })).toThrow();
    expect(() => schema.parse({ limit: true })).toThrow();
    expect(() => schema.parse({ offset: "0" })).toThrow();
    expect(schema.parse({ limit: 5, offset: 5 })).toMatchObject({ limit: 5, offset: 5 });
  });

  it("reads an explicit null on a filter as not supplied, and refuses it where Python has no None", () => {
    const schema = z.object(listJobsOp.input);
    const parsed = schema.parse({ historyId: null, state: null, dateRangeMin: null, dateRangeMax: null });
    expect(parsed.historyId).toBeNull();
    expect(parsed.state).toBeNull();
    // order_by, view, limit and offset are plain defaulted parameters there, with no null branch.
    expect(() => schema.parse({ orderBy: null })).toThrow();
    expect(() => schema.parse({ view: null })).toThrow();
    expect(() => schema.parse({ limit: null })).toThrow();
    expect(() => schema.parse({ offset: null })).toThrow();
  });
});
