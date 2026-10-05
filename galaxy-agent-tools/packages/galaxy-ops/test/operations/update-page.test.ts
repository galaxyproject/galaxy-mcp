import { describe, it, expect, vi } from "vitest";
import { updatePageOp, updatePage } from "../../src/operations/update-page";
import { runWithEnvelope } from "../../src/operations/registry";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyConnectionError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("update_page", () => {
  it("is a write op", () => {
    expect(updatePageOp.readOnly).toBe(false);
  });

  it("attributes the new revision to the agent and drops the rendered form", async () => {
    const client = mockClient({
      PUT: (path, init) => {
        expect(path).toBe("/api/pages/{id}");
        expect(init.params.path.id).toBe("page1");
        expect(init.body.edit_source).toBe("agent");
        expect(init.body.content).toBe("# updated");
        expect(init.body.title).toBeUndefined();
        return {
          data: { id: "page1", title: "Notebook 1", content: "<rendered>", content_editor: "# updated" },
          response: { status: 200 },
        };
      },
    });
    const out = await updatePage({ pageId: "page1", content: "# updated" }, ctxWith(client));
    expect(out.content_editor).toBe("# updated");
    expect("content" in out).toBe(false);
  });

  it("sends a title-only edit", async () => {
    const client = mockClient({
      PUT: (_path, init) => {
        expect(init.body.title).toBe("Renamed");
        expect(init.body.content).toBeUndefined();
        return { data: { id: "page1", title: "Renamed" }, response: { status: 200 } };
      },
    });
    await updatePage({ pageId: "page1", title: "Renamed" }, ctxWith(client));
  });

  // The other server has no refusal here: an edit with nothing in it is a PUT carrying only
  // its edit_source, which Galaxy answers. update_history does refuse its empty update, and
  // copying that sibling's rule into this tool would be a rule one surface has.
  it("sends an edit that changes nothing, because the other server sends it", async () => {
    const PUT = vi.fn(() => ({ data: { id: "page1" }, response: { status: 200 } }));
    await updatePage({ pageId: "page1" }, ctxWith(mockClient({ PUT })));
    expect(PUT).toHaveBeenCalledOnce();
    expect(PUT.mock.calls[0]?.[1]?.body).toEqual({ edit_source: "agent" });
  });

  it("envelopes a missing page as not_found", async () => {
    const client = mockClient({ PUT: () => ({ error: { err_msg: "gone" }, response: { status: 404 } }) });
    const r = await runWithEnvelope(updatePageOp as any, { pageId: "nope", content: "x" }, ctxWith(client));
    expect(r.success).toBe(false);
    expect(r.errorKind).toBe("not_found");
  });
});

describe("update_page section edits and expectHash", () => {
  const DOC = "## Record\n\nintro\n\n## Methods\n\nold\n\n## Results\n\nfindings\n";

  /** A page holding DOC that records every PUT body, and every route asked. */
  function galaxy(doc = DOC, contentFormat = "markdown") {
    const puts: any[] = [];
    const asked: string[] = [];
    const client = mockClient({
      GET: (path) => {
        asked.push(`GET ${path}`);
        return {
          data: { id: "page1", content_format: contentFormat, content_editor: doc, content: "<render>" },
          response: { status: 200 },
        };
      },
      PUT: (path, init) => {
        asked.push(`PUT ${path}`);
        puts.push(init.body);
        return { data: { id: "page1", content_editor: init.body.content ?? doc }, response: { status: 200 } };
      },
    });
    return { ctx: ctxWith(client), puts, asked };
  }

  it("replaces one section by its heading and leaves the others alone", async () => {
    const g = galaxy();
    await updatePage({ pageId: "page1", sectionHeading: "## Methods", sectionContent: "## Methods\n\nnew\n" }, g.ctx);
    expect(g.puts[0].content).toBe("## Record\n\nintro\n\n## Methods\n\nnew\n\n## Results\n\nfindings\n");
  });

  it("refuses a section edit on an HTML page rather than appending markdown to it", async () => {
    const g = galaxy("<p>body</p>", "html");
    await expect(
      updatePage({ pageId: "page1", sectionHeading: "## Notes", sectionContent: "## Notes\n\nadded" }, g.ctx),
    ).rejects.toThrow("Page 'page1' is authored as html; a section can be replaced only in a markdown page");
    expect(g.puts).toEqual([]);
  });

  it("appends a section whose heading the page does not have, as Galaxy's editor does", async () => {
    const g = galaxy();
    await updatePage({ pageId: "page1", sectionHeading: "## Notes", sectionContent: "## Notes\n\nadded" }, g.ctx);
    expect(g.puts[0].content.endsWith("findings\n\n## Notes\n\nadded")).toBe(true);
  });

  it("answers with the page's new content_hash and without the render", async () => {
    const g = galaxy();
    const out = await updatePage({ pageId: "page1", content: "raw markdown" }, g.ctx);
    expect(out.content_hash).toBe("f2285a32");
    expect("content" in out).toBe(false);
    expect(g.asked).toEqual(["PUT /api/pages/{id}"]);
  });

  it("writes when expectHash is the page's hash", async () => {
    const g = galaxy("raw markdown");
    await updatePage({ pageId: "page1", content: "fresh", expectHash: "f2285a32" }, g.ctx);
    expect(g.puts).toEqual([{ edit_source: "agent", content: "fresh" }]);
  });

  it("fails without writing when the page changed since it was read", async () => {
    const g = galaxy("raw markdown");
    const env = await runWithEnvelope(updatePageOp, { pageId: "page1", content: "x", expectHash: "deadbeef" } as never, g.ctx);
    expect(env.success).toBe(false);
    expect((env as any).message).toBe(
      "Page 'page1' changed since it was read: its content_hash is now f2285a32, not deadbeef. " +
        "Read it again with get_page and make the edit against what is there now.",
    );
    expect(g.puts).toEqual([]);
  });

  it("refuses half a section edit, or a section and content together, before asking Galaxy", async () => {
    for (const input of [
      { sectionHeading: "## Methods" },
      { sectionContent: "## Methods\n\nnew" },
      { content: "x", sectionHeading: "## Methods", sectionContent: "## Methods" },
    ]) {
      const g = galaxy();
      const env = await runWithEnvelope(updatePageOp, { pageId: "page1", ...input } as never, g.ctx);
      expect(env.success, JSON.stringify(input)).toBe(false);
      expect(g.asked).toEqual([]);
    }
  });

  it("refuses a directive naming an object by something that is not an encoded id", async () => {
    const g = galaxy();
    const content =
      "about history_id=3 in prose\n```galaxy\nhistory_dataset_display(history_dataset_id=reads)\n```\n" +
      "${galaxy job_metrics(job_id=12)} ${galaxy history_dataset_name(history_dataset_id=0c97fda4aafcf418)}";
    const env = await runWithEnvelope(updatePageOp, { pageId: "page1", content } as never, g.ctx);
    expect(env.success).toBe(false);
    expect((env as any).message).toContain("history_dataset_id=reads, job_id=12.");
    expect((env as any).message).not.toContain("history_id=3");
    expect(g.asked).toEqual([]);
  });

  it("takes an encoded id of any length Galaxy's cipher produces", async () => {
    const g = galaxy();
    const long = "0c97fda4aafcf418".repeat(2);
    await updatePage({ pageId: "page1", content: `\`\`\`galaxy\ninvocation_outputs(invocation_id=${long})\n\`\`\`` }, g.ctx);
    expect(g.puts).toHaveLength(1);
  });
});
