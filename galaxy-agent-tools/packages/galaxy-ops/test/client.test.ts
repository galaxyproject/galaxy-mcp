import { describe, it, expect, vi } from "vitest";
import { createGalaxyClient } from "../src/client";

describe("createGalaxyClient", () => {
  it("sends x-api-key and resolves baseUrl", async () => {
    const fetchSpy = vi.fn(async () =>
      new Response(JSON.stringify({ id: "1", username: "u", email: "e@x" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    const client = createGalaxyClient("https://galaxy.example", "KEY123", fetchSpy);
    const { data, error } = await client.GET("/api/users/{user_id}", {
      params: { path: { user_id: "current" } },
    });
    expect(error).toBeUndefined();
    expect((data as any).username).toBe("u");

    // openapi-fetch 0.17 calls fetch(request, requestInitExt) where request is a
    // Request object with headers already baked in -- not (url, init) as in older
    // versions. Read from the Request object directly.
    const [reqOrUrl, init] = fetchSpy.mock.calls[0]!;
    if (reqOrUrl instanceof Request) {
      // openapi-fetch 0.17 path: Request object carries url + headers
      expect(reqOrUrl.url).toBe("https://galaxy.example/api/users/current");
      expect(reqOrUrl.headers.get("x-api-key")).toBe("KEY123");
    } else {
      // fallback: old (url, init) style -- still verify the same invariants
      expect(String(reqOrUrl)).toBe("https://galaxy.example/api/users/current");
      expect((init!.headers as Headers).get("x-api-key")).toBe("KEY123");
    }
  });
});
