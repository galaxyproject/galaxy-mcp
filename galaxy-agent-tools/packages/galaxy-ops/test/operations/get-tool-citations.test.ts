import { describe, it, expect } from "vitest";
import { getToolCitationsOp, getToolCitations } from "../../src/operations/get-tool-citations";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_tool_citations", () => {
  it("fetches tool show payload and maps name/version/citations", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/tools/{tool_id}");
        expect(init.params.path.tool_id).toBe("fastqc");
        expect(init.params.query.io_details).toBe(false);
        expect(init.params.query.link_details).toBe(false);
        return {
          data: { id: "fastqc", name: "FastQC", version: "0.74", citations: [{ type: "doi", value: "10.1/foo" }] },
          response: { status: 200 },
        };
      },
    });
    const out = await getToolCitations({ toolId: "fastqc" }, ctxWith(client));
    expect(out.tool_name).toBe("FastQC");
    expect(out.tool_version).toBe("0.74");
    expect(out.citations).toHaveLength(1);
  });

  it("defaults citations to [] when missing from payload", async () => {
    const client = mockClient({
      GET: () => ({ data: { id: "cat1", name: "Concatenate" }, response: { status: 200 } }),
    });
    const out = await getToolCitations({ toolId: "cat1" }, ctxWith(client));
    expect(out.citations).toEqual([]);
  });

  it("project message counts the citations and names the tool that was asked for", () => {
    // Not the name Galaxy answered with, which is in data: the other server's
    // sentence quotes the id, so a caller reading it knows what to ask again with.
    const result = { tool_name: "FastQC", tool_version: "0.74", citations: [{}] };
    const msg = getToolCitationsOp.project!(result as any, { toolId: "fastqc" });
    expect(msg.message).toBe("Retrieved 1 citations for tool 'fastqc'");
  });

  it("writes the plural for one citation, as the other server does", () => {
    const result = { tool_name: "FastQC", citations: [{}] };
    expect(getToolCitationsOp.project!(result as any, { toolId: "cat1" }).message).toBe(
      "Retrieved 1 citations for tool 'cat1'",
    );
  });
});
