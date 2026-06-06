// Re-export upstream-generated types under stable names. Mirrors Galaxy's api-client
// (client/packages/api-client/src/schema/index.ts): paths is aliased to GalaxyApiPaths.
import type { components, paths as GalaxyApiPaths } from "./schema";
export type { components, GalaxyApiPaths };
