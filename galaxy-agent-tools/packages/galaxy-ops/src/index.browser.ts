// The surface a browser can run; the package's "browser" condition resolves here.
import "./operations/all-browser";
export * from "./context";
export * from "./errors";
export {
  parseGalaxyVersion,
  parseRequirement,
  requirementSentence,
  satisfiesRequirement,
  type GalaxyVersion,
  type VersionRequirement,
} from "./version";
export { isJobTerminal, isJobSuccess } from "./terminal-states";
export type { Operation, OperationDomain, GalaxyResult, Pagination, InputOf, AnyOperation } from "./operations/types";
export {
  allOperations,
  assertVersionSupported,
  describeOperation,
  runOperation,
  runWithEnvelope,
} from "./operations/registry";
export { validatePagination, paginate, paginationInfo, OUTPUT_BUDGET_BYTES } from "./operations/pagination";
export { laxenLikePydantic, isPlainObject } from "./laxen";
export type { PaginationInfo, Paged } from "./operations/pagination";
export { getUserOp, getUser, type CurrentUser } from "./operations/get-user";
export { runToolOp, runTool } from "./operations/run-tool";
export type { ToolRun, ToolInputs, ImplicitCollectionRef } from "./execute-tool-request";
export {
  getInvocationsOp,
  getInvocations,
  type InvocationDetail,
  type InvocationSummary,
  type GetInvocationsResult,
} from "./operations/get-invocations";
export {
  getServerInfoOp,
  getServerInfo,
  type ServerInfo,
  type UnsupportedOp,
} from "./operations/get-server-info";
export { getHistoriesOp, getHistories, type Histories } from "./operations/get-histories";
export { listHistoryIdsOp, listHistoryIds, type HistoryRef } from "./operations/list-history-ids";
export { getHistoryDetailsOp, getHistoryDetails, type HistoryDetail } from "./operations/get-history-details";
export { createHistoryOp, createHistory, type CreatedHistory } from "./operations/create-history";
export { getDatasetDetailsOp, getDatasetDetails, type DatasetDetail } from "./operations/get-dataset-details";
export { getCollectionDetailsOp, getCollectionDetails, type CollectionDetail } from "./operations/get-collection-details";
export { getHistoryContentsOp, getHistoryContents, type HistoryContents } from "./operations/get-history-contents";
export { listWorkflowsOp, listWorkflows, type Workflows } from "./operations/list-workflows";
export { getWorkflowDetailsOp, getWorkflowDetails, type WorkflowDetail } from "./operations/get-workflow-details";
export { getToolDetailsOp, getToolDetails, type ToolDetail } from "./operations/get-tool-details";
export { searchToolsByNameOp, searchToolsByName, type ToolListItem } from "./operations/search-tools-by-name";
export {
  getToolPanelOp,
  getToolPanel,
  type ToolPanelResult,
  type ToolPanelOverview,
  type ToolPanelSection,
  type PanelEntry,
  type SlimTool,
} from "./operations/get-tool-panel";
export { getToolCitationsOp, getToolCitations, type ToolCitationsResult } from "./operations/get-tool-citations";
export { getToolRunExamplesOp, getToolRunExamples, type ToolRunExamples } from "./operations/get-tool-run-examples";
export { searchToolsByKeywordsOp, searchToolsByKeywords, type ToolKeywordMatch } from "./operations/search-tools-by-keywords";
export { getJobDetailsOp, getJobDetails, type GetJobDetailsResult, type JobDetail } from "./operations/get-job-details";
export { updateHistoryOp, updateHistory, type UpdatedHistory } from "./operations/update-history";
export { cancelWorkflowInvocationOp, cancelWorkflowInvocation, type CancelledInvocation } from "./operations/cancel-workflow-invocation";
export { getIwcWorkflowsOp, getIwcWorkflows } from "./operations/get-iwc-workflows";
export { getIwcWorkflowDetailsOp, getIwcWorkflowDetails, type IwcWorkflowDetail } from "./operations/get-iwc-workflow-details";
export { searchIwcWorkflowsOp, searchIwcWorkflows } from "./operations/search-iwc-workflows";
export { recommendIwcWorkflowsOp, recommendIwcWorkflows, type Recommendations } from "./operations/recommend-iwc-workflows";
export { importWorkflowFromIwcOp, importWorkflowFromIwc, type ImportedWorkflow } from "./operations/import-workflow-from-iwc";
export { listUserToolsOp, listUserTools, type UserTool } from "./operations/list-user-tools";
export { createUserToolOp, createUserTool, type CreatedUserTool } from "./operations/create-user-tool";
export { deleteUserToolOp, deleteUserTool, type DeletedUserTool } from "./operations/delete-user-tool";
export { runUserToolOp, runUserTool, type UserToolRun } from "./operations/run-user-tool";
export { getToolInputTemplateOp, getToolInputTemplate, type ToolInputTemplateResult } from "./operations/get-tool-input-template";
export { getWorkflowInputTemplateOp, getWorkflowInputTemplate, resolveWorkflowSlots, type ResolvedSlots } from "./operations/get-workflow-input-template";
export { invokeWorkflowOp, invokeWorkflow, getDatatypesMapping, type InvocationResult } from "./operations/invoke-workflow";
export { uploadFileFromUrlOp, uploadFileFromUrl, type UploadFileFromUrlResult } from "./operations/upload-file-from-url";
export { listPagesOp, listPages } from "./operations/list-pages";
export { getPageOp, getPage } from "./operations/get-page";
export { createPageOp, createPage } from "./operations/create-page";
export { updatePageOp, updatePage } from "./operations/update-page";
export { listPageRevisionsOp, listPageRevisions } from "./operations/list-page-revisions";
export { getPageRevisionOp, getPageRevision } from "./operations/get-page-revision";
export { revertPageRevisionOp, revertPageRevision } from "./operations/revert-page-revision";
export type {
  PageSummary,
  PageDetail,
  PageRevisionSummary,
  PageRevisionDetails,
  ContentEditorSource,
} from "./operations/pages-common";
export type { WorkflowInputTemplate, WorkflowSlot, WorkflowGuide, DatatypesMapping, ValidationReport } from "./workflow-inputs";
export { validateInputs, subtypeSatisfies } from "./workflow-inputs";
export type { IwcWorkflow, EnrichedIwcWorkflow } from "./iwc-manifest";
