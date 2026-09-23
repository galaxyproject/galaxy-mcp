import { describe, it, expect } from "vitest";
import * as api from "../../src/index";
import { allOperations } from "../../src/operations/registry";
import { GalaxyVersionError } from "../../src/errors";
import { parseGalaxyVersion } from "../../src/version";
import type { GalaxyContext } from "../../src/context";
import { mockClient } from "../util/mock-client";

/**
 * The Pages ops 26.0 genuinely cannot serve. `get_page` is deliberately NOT here: 26.0's
 * PageDetails already carries content_editor, so it works there and gating it would take
 * away something that runs. `list_pages` IS here even though the endpoint is older, because
 * 26.0 ignores the history_id filter and answers with every page the user can see -- a
 * refusal beats silently handing back the wrong set.
 */
const GATED = [
  "list_pages",
  "create_page",
  "update_page",
  "list_page_revisions",
  "get_page_revision",
  "revert_page_revision",
];
const UNGATED_PAGES = ["get_page"];

const gated = allOperations.filter((op) => op.requires);

/** A 26.0 server, and a client that records anything an unguarded op would send. */
function oldGalaxy() {
  const sent: string[] = [];
  const record = (path: string) => {
    sent.push(path);
    return { data: {}, response: { status: 200 } };
  };
  const ctx: GalaxyContext = {
    client: mockClient({ GET: record, POST: record, PUT: record, DELETE: record }),
    poll: { intervalMs: 1, maxIntervalMs: 1, backoff: 1, jitter: 0, timeoutMs: 1 },
    galaxyVersion: async () => ({ version: parseGalaxyVersion("26.0"), source: "server" as const }),
  };
  return { ctx, sent };
}

const camel = (name: string) => name.replace(/_(\w)/g, (_, c: string) => c.toUpperCase());

describe("every op's summary", () => {
  it("ends in terminal punctuation, which the appended requirement sentence assumes", () => {
    // describeOperation joins with a bare space; a summary without a full stop would run
    // straight into "Requires Galaxy ...".
    const unterminated = allOperations.filter((op) => !/[.!?]$/.test(op.summary));
    expect(unterminated.map((op) => op.name)).toEqual([]);
  });
});

describe("the ops that need a newer Galaxy", () => {
  it("is exactly the ops a 26.0 server cannot serve", () => {
    expect(gated.map((op) => op.name).sort()).toEqual([...GATED].sort());
  });

  it("leaves alone the Pages ops that do work on 26.0", () => {
    for (const name of UNGATED_PAGES) {
      const op = allOperations.find((o) => o.name === name);
      expect(op, `${name} is not registered`).toBeDefined();
      expect(op?.requires, `${name} should not be gated`).toBeUndefined();
    }
  });

  it("asks for 26.1, where the pages revision API arrived", () => {
    for (const op of gated) expect(op.requires?.galaxy).toBe(">=26.1");
  });

  it("refuses when called on the op object itself, which anyone can reach", async () => {
    // createPageOp.run(...) is the escape the named export cannot cover.
    for (const op of gated) {
      const { ctx, sent } = oldGalaxy();
      await expect(
        (op.run as (i: unknown, c: GalaxyContext) => Promise<unknown>)({}, ctx),
        `${op.name}.run was not guarded`,
      ).rejects.toBeInstanceOf(GalaxyVersionError);
      expect(sent, `${op.name} sent a request it should have refused`).toEqual([]);
    }
  });

  it("refuses through the direct export too, not only through a surface", async () => {
    for (const op of gated) {
      const direct = (api as Record<string, unknown>)[camel(op.name)];
      expect(typeof direct, `${op.name} has no direct export named ${camel(op.name)}`).toBe(
        "function",
      );
      const call = direct as (i: unknown, c: GalaxyContext) => Promise<unknown>;
      const { ctx, sent } = oldGalaxy();
      await expect(call({}, ctx)).rejects.toBeInstanceOf(GalaxyVersionError);
      expect(sent, `${op.name} sent a request it should have refused`).toEqual([]);
    }
  });
});
