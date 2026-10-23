import { describe, expect, it } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createGalaxyContext } from "@galaxyproject/galaxy-ops";
import { buildServer } from "../src/server";

// Nothing here reaches Galaxy: a call that gets that far fails at this fetch.
const ctx = createGalaxyContext({
  baseUrl: "http://galaxy.invalid",
  apiKey: "k",
  fetchImpl: async () => {
    throw new Error("no Galaxy in this test");
  },
});

async function connect(allowLocalFiles: boolean): Promise<Client> {
  const [clientSide, serverSide] = InMemoryTransport.createLinkedPair();
  await buildServer(ctx, { allowLocalFiles }).connect(serverSide);
  const client = new Client({ name: "local-files-test", version: "0" });
  await client.connect(clientSide);
  return client;
}

const text = (result: Awaited<ReturnType<Client["callTool"]>>) => JSON.stringify(result.content);

describe("tools that take a path on the server", () => {
  it("are refused with the Python server's sentence when local files are off", async () => {
    const client = await connect(false);
    const upload = await client.callTool({ name: "upload_file", arguments: { path: "/etc/hosts" } });
    expect(upload.isError).toBe(true);
    expect(upload.content).toEqual([
      {
        type: "text",
        text:
          "Error calling tool 'upload_file': upload_file is disabled over HTTP transports, because the path " +
          "would refer to the MCP server's filesystem rather than yours. Use upload_file_from_url to load " +
          "data the Galaxy server can fetch itself. Operators of a trusted single-user deployment can " +
          "re-enable it with GALAXY_MCP_ALLOW_LOCAL_FILES=1.",
      },
    ]);
    const download = await client.callTool({ name: "download_dataset", arguments: { dataset_id: "d1", file_path: "/tmp/out" } });
    expect(download.isError).toBe(true);
    expect(text(download)).toContain("download_dataset(file_path=...) is disabled over HTTP");
  });

  it("let an empty file_path through, as Python does, since it writes nothing", async () => {
    const client = await connect(false);
    const result = await client.callTool({ name: "download_dataset", arguments: { dataset_id: "d1", file_path: "" } });
    expect(text(result)).not.toContain("disabled over HTTP");
  });

  it("are not refused when local files are on, as over stdio", async () => {
    const client = await connect(true);
    const result = await client.callTool({ name: "upload_file", arguments: { path: "/no/such/file" } });
    expect(text(result)).not.toContain("disabled over HTTP");
  });
});
