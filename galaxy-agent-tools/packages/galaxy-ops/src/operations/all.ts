// The single import-for-side-effect list. Importing this guarantees every op is
// registered, regardless of whether index.ts happens to re-export it. New ops MUST
// add a line here.
import "./get-user";
import "./run-tool";
import "./get-invocations";
import "./get-server-info";
import "./get-histories";
import "./list-history-ids";
import "./get-history-details";
import "./create-history";
import "./get-dataset-details";
import "./get-collection-details";
import "./get-history-contents";
import "./list-workflows";
import "./get-workflow-details";
import "./get-tool-details";
