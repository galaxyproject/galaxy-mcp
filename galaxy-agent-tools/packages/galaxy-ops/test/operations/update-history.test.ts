import { describe, it, expect } from "vitest";
import { updateHistoryOp, updateHistory } from "../../src/operations/update-history";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";
import { GalaxyValidationError } from "../../src/errors";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("update_history", () => {
  it("is a write op", () => {
    expect(updateHistoryOp.readOnly).toBe(false);
  });

  it("sends only provided fields in PUT body", async () => {
    const client = mockClient({
      PUT: (path, init) => {
        expect(path).toBe("/api/histories/{history_id}");
        expect(init.params.path.history_id).toBe("h1");
        // only name should be in the body, not annotation
        expect(init.body.name).toBe("New Name");
        expect(init.body.annotation).toBeUndefined();
        return { data: { id: "h1", name: "New Name" }, response: { status: 200 } };
      },
    });
    const out = await updateHistory({ historyId: "h1", name: "New Name" }, ctxWith(client));
    expect((out as any).id).toBe("h1");
  });

  // The other server refuses this before it even checks the connection, and names the fields
  // it would have taken. Its kind is `validation` and not `connection`: nothing was reached,
  // and an agent told "connection" backs off and retries a call that cannot ever work.
  it("refuses with the other server's sentence when there is no field to update", async () => {
    const client = mockClient({
      PUT: () => ({ data: {}, response: { status: 200 } }),
    });
    await expect(updateHistory({ historyId: "h1" }, ctxWith(client))).rejects.toThrow(
      "No fields provided to update. Pass at least one of: name, annotation, tags, deleted, published.",
    );
    await expect(updateHistory({ historyId: "h1" }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyValidationError,
    );
  });

  /**
   * A null is an unset field, and the sentence has to agree with the body.
   *
   * The body already dropped one -- `!= null` -- but the message counted anything that was
   * not `undefined`, so a caller spelling "leave this alone" as `published: null` was told
   * published had been updated by the same call that did not send it. The other server builds
   * both out of one dict, so the two cannot disagree over there.
   */
  it("names only the fields it sent, so a null is not reported as updated", () => {
    expect(
      updateHistoryOp.project!({} as never, { historyId: "h1", name: "new", published: null }),
    ).toEqual({ message: "Updated history h1 (name)" });
  });

  it("names none of the five when every one of them is null", () => {
    // The refusal is what a caller actually gets here; this pins the projection alone, so a
    // future field cannot be counted as updated on the strength of being mentioned.
    expect(
      updateHistoryOp.project!({} as never, {
        historyId: "h1",
        name: null,
        annotation: null,
        tags: null,
        deleted: null,
        published: null,
      }),
    ).toEqual({ message: "Updated history h1 ()" });
  });

  it("names all five when all five are set, in the other server's order", () => {
    expect(
      updateHistoryOp.project!({} as never, {
        historyId: "h1",
        published: false,
        deleted: false,
        tags: [],
        annotation: "",
        name: "new",
      }),
    ).toEqual({ message: "Updated history h1 (name, annotation, tags, deleted, published)" });
  });

  it("sends tags and deleted when provided", async () => {
    const client = mockClient({
      PUT: (_path, init) => {
        expect(init.body.tags).toEqual(["tag1", "tag2"]);
        expect(init.body.deleted).toBe(true);
        return { data: { id: "h2" }, response: { status: 200 } };
      },
    });
    await updateHistory({ historyId: "h2", tags: ["tag1", "tag2"], deleted: true }, ctxWith(client));
  });
});
