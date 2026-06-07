// THE single bindings source: Galaxy's published OpenAPI bindings on npm.
// galaxy-ops consumes the TYPES only (components, GalaxyApiPaths) and builds its own
// openapi-fetch client (see client.ts). Bump @galaxyproject/galaxy-api-client per the
// targeted Galaxy release (currently 26.0.x; move to 26.1.x once it publishes).
export type { components, GalaxyApiPaths } from "@galaxyproject/galaxy-api-client";

import type { components } from "@galaxyproject/galaxy-api-client";
export type Schemas = components["schemas"];
