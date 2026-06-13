import { describe, it, expect } from "vitest";
import { createGalaxyClient } from "../../src/client";

const URL = process.env.GALAXY_URL;
const KEY = process.env.GALAXY_API_KEY;
const run = URL && KEY ? describe : describe.skip;

run("integration: whoami against a real Galaxy", () => {
  it("GET current user returns id/email/username", async () => {
    const client = createGalaxyClient(URL!, KEY!);
    const { data, error, response } = await client.GET("/api/users/{user_id}", {
      params: { path: { user_id: "current" } },
    });
    expect(error, `HTTP ${response.status}`).toBeUndefined();
    expect((data as any).id).toBeTruthy();
    expect((data as any).email).toContain("@");
  });
});
