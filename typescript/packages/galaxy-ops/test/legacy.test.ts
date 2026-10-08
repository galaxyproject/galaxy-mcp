import { describe, it, expect } from "vitest";
import { legacyGet, legacyPost, legacyDelete } from "../src/legacy";
import { createGalaxyClient } from "../src/client";
import { mockClient } from "./util/mock-client";
import { DEFAULT_POLL } from "../src/context";
import { GalaxyAuthError, GalaxyNotFoundError } from "../src/errors";
import type { GalaxyContext } from "../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("legacyGet", () => {
  it("returns the data for an off-schema path via the configured client", async () => {
    const client = mockClient({
      GET: (path) => {
        expect(path).toBe("/api/tools/{tool_id}");
        return { data: { id: "cat1", name: "Concatenate" }, response: { status: 200 } };
      },
    });
    const out = await legacyGet<{ id: string }>(ctxWith(client), "/api/tools/{tool_id}", {
      params: { path: { tool_id: "cat1" } },
    });
    expect(out.id).toBe("cat1");
  });

  it("classifies HTTP failures into typed errors", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "no" }, response: { status: 404 } }) });
    await expect(legacyGet(ctxWith(client), "/api/tools/{tool_id}", {})).rejects.toBeInstanceOf(GalaxyNotFoundError);
  });
});

describe("legacyPost", () => {
  it("returns data for an off-schema POST and classifies failures", async () => {
    const ok = mockClient({
      POST: (path, init) => {
        expect(path).toBe("/api/unprivileged_tools");
        expect(init.body.src).toBe("representation");
        return { data: { id: "t1", uuid: "u1" }, response: { status: 200 } };
      },
    });
    const out = await legacyPost<{ uuid: string }>(ctxWith(ok), "/api/unprivileged_tools", {
      body: { src: "representation" },
    });
    expect(out.uuid).toBe("u1");

    const bad = mockClient({ POST: () => ({ error: { err_msg: "no" }, response: { status: 404 } }) });
    await expect(legacyPost(ctxWith(bad), "/api/unprivileged_tools", {})).rejects.toBeInstanceOf(
      GalaxyNotFoundError,
    );
  });
});

describe("legacyDelete", () => {
  it("returns data for an off-schema DELETE and classifies failures", async () => {
    const ok = mockClient({
      DELETE: (path) => {
        expect(path).toBe("/api/unprivileged_tools/{uuid}");
        return { data: { uuid: "u1", deactivated: true }, response: { status: 200 } };
      },
    });
    const out = await legacyDelete<{ deactivated: boolean }>(ctxWith(ok), "/api/unprivileged_tools/{uuid}", {
      params: { path: { uuid: "u1" } },
    });
    expect(out.deactivated).toBe(true);

    const bad = mockClient({ DELETE: () => ({ error: { err_msg: "no" }, response: { status: 404 } }) });
    await expect(legacyDelete(ctxWith(bad), "/api/unprivileged_tools/{uuid}", {})).rejects.toBeInstanceOf(
      GalaxyNotFoundError,
    );
  });

  it("treats a 204 No Content (null body) as success, not an error", async () => {
    const noContent = mockClient({ DELETE: () => ({ data: undefined, response: { status: 204 } }) });
    await expect(
      legacyDelete(ctxWith(noContent), "/api/unprivileged_tools/{uuid}", {}),
    ).resolves.toBeUndefined();
  });
});

/**
 * A reply with an empty body, which is the case the truthiness guard got wrong.
 *
 * openapi-fetch reads a failed reply's body as text and hands it back parsed if it can
 * be: for an empty body that is the empty string, which is falsy, so a guard asking
 * whether there is an error body answered "no" for a 403. Only the status can decide,
 * and these go through the real client rather than a stand-in because the stand-in is
 * what hid it -- it hands back whatever a test says, and no test says `error: ""`.
 */
describe("a reply with an empty body", () => {
  const canned = (status: number) =>
    createGalaxyClient("https://galaxy.example", "not-a-key", async () =>
      // 204 is the one status that carries no body at all; the rest carry an empty one.
      status === 204
        ? new Response(null, { status })
        : new Response("", { status, headers: { "content-type": "application/json" } }),
    );

  it("is a failure on a 403, not a success", async () => {
    const err = await legacyDelete(
      ctxWith(canned(403)),
      "/api/unprivileged_tools/{uuid}",
      {},
    ).then(
      () => null,
      (e: unknown) => e,
    );
    expect(err, "a 403 with no body answered success").toBeInstanceOf(GalaxyAuthError);
    expect((err as GalaxyAuthError).http?.status).toBe(403);
  });

  it("is a failure on a 404, not a success", async () => {
    await expect(
      legacyDelete(ctxWith(canned(404)), "/api/unprivileged_tools/{uuid}", {}),
    ).rejects.toBeInstanceOf(GalaxyNotFoundError);
  });

  it("is still a success on a 204, which is what a DELETE answers with", async () => {
    await expect(
      legacyDelete(ctxWith(canned(204)), "/api/unprivileged_tools/{uuid}", {}),
    ).resolves.toBeUndefined();
  });

  it("is a failure on a 403 for the other two verbs too", async () => {
    await expect(
      legacyGet(ctxWith(canned(403)), "/api/unprivileged_tools", {}),
    ).rejects.toBeInstanceOf(GalaxyAuthError);
    await expect(
      legacyPost(ctxWith(canned(403)), "/api/unprivileged_tools", {}),
    ).rejects.toBeInstanceOf(GalaxyAuthError);
  });
});
