import { describe, expect, it } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createGalaxyContext } from "@galaxyproject/galaxy-ops";
import * as library from "../src/library";
import pkg from "../package.json";

describe("the library entry", () => {
  it("is what the package exports, beside the command", () => {
    expect(pkg.exports["."]).toEqual({ types: "./dist/library.d.ts", import: "./dist/library.js" });
    expect(pkg.bin["galaxy-mcp"]).toBe("./dist/index.js");
  });

  it("builds a server without starting one", async () => {
    const ctx = createGalaxyContext({ baseUrl: "http://galaxy.invalid", apiKey: "k" });
    const [clientSide, serverSide] = InMemoryTransport.createLinkedPair();
    await library.buildServer(ctx).connect(serverSide);
    const client = new Client({ name: "library-test", version: "0" });
    await client.connect(clientSide);
    expect((await client.listTools()).tools.map((tool) => tool.name)).toEqual(library.toolNames());
    expect(typeof library.createHttpHandler).toBe("function");
  });
});
