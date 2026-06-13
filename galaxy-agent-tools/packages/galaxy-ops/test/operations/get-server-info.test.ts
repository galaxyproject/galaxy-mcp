import { describe, it, expect } from "vitest";
import { getServerInfoOp, getServerInfo } from "../../src/operations/get-server-info";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, baseUrl: "https://g.example", poll: DEFAULT_POLL });

describe("get_server_info", () => {
  it("has parity name and returns url + version + config", async () => {
    expect(getServerInfoOp.name).toBe("get_server_info");
    const client = mockClient({
      GET: (path) => {
        if (path === "/api/version") return { data: { version_major: "26.0" }, response: { status: 200 } };
        if (path === "/api/configuration") return { data: { brand: "Test" }, response: { status: 200 } };
        return { error: "x", response: { status: 500 } };
      },
    });
    const out = await getServerInfo({}, ctxWith(client));
    expect(out.url).toBe("https://g.example");
    expect((out.version as any).version_major).toBe("26.0");
    expect((out.config as any).brand).toBe("Test");
  });
});
