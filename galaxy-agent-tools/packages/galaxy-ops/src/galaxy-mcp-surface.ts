/**
 * What galaxy-mcp's Python server advertises for each of its tools, as an MCP client receives it:
 * the description, with any Galaxy requirement in it, and each parameter's description, in the
 * order the tool declares them ("" where it has none). For a client that offers these tools under
 * galaxy-mcp's names and wants its model told what that server's is.
 *
 * GENERATED FILE -- do not edit. Regenerate from packages/galaxy-ops with
 * `node scripts/generate-galaxy-mcp-surface.mjs`. Its input is
 * mcp-server-galaxy-py/tests/testdata/mcp-surface.json, which that server writes with
 * `uv run python -m tests.surface_manifest`, and `test/galaxy-mcp-surface.test.ts` compares this
 * file to that one, so the two copies cannot drift.
 */
export interface GalaxyMcpTool {
  readonly description: string;
  readonly parameters: Readonly<Record<string, string>>;
}

export const GALAXY_MCP_SURFACE: Readonly<Record<string, GalaxyMcpTool>> = {
  "cancel_workflow_invocation": {
    description: "Cancel a running workflow invocation",
    parameters: {
      "invocation_id": "ID of the workflow invocation to cancel - a hexadecimal hash string",
    },
  },
  "connect": {
    description: "Connect to Galaxy server",
    parameters: {
      "url": "Galaxy server URL (optional, uses GALAXY_URL env var if not provided)",
      "api_key": "Galaxy API key (optional, uses GALAXY_API_KEY env var if not provided)",
    },
  },
  "create_history": {
    description: "Create a new history to organize datasets and analyses.\n\nA history is the primary workspace in Galaxy. Create a new history for each\ndistinct project or analysis to keep your work organized.\n\nRECOMMENDED WORKFLOW:\n1. Create a history with a descriptive name\n2. Upload your input data: upload_file() or upload_file_from_url()\n3. Run tools on the data: run_tool()\n4. View results: get_history_contents()",
    parameters: {
      "history_name": "Descriptive name for the history.\n          Best practices:\n          - Include project/sample name: \"RNA-seq Sample A\"\n          - Include date if relevant: \"ChIP-seq 2024-01\"\n          - Be specific: \"BWA alignment of patient_001\"",
    },
  },
  "create_page": {
    description: "Create a markdown page (Galaxy Notebook or Report).\n\nRequires Galaxy 26.1 or newer.\n\nAttaching a page to a history is what needs the newer server, and the whole\ntool is refused rather than only that call: an older one has no history_id\non the create payload, and answers with a summary carrying no editable\ncontent whichever kind you asked for.\n\nPass history_id to create a history-attached notebook (title auto-fills\nfrom the history if omitted). Omit history_id to create a standalone\nreport -- reports REQUIRE both a title and a unique slug\n(lowercase/digits/hyphens).\n\nContent is Galaxy-flavored markdown. To embed a dataset, use a directive\nwith the ENCODED dataset id, e.g.\n`history_dataset_display(history_dataset_id=<encoded-dataset-id>)` or\n`history_dataset_collection_display(history_dataset_collection_id=...)`.\nGet encoded ids from get_history_contents / get_dataset_details.",
    parameters: {
      "history_id": "Encoded history id to attach the page to (notebook).",
      "title": "Page title.",
      "content": "Initial markdown content.",
      "annotation": "Optional annotation attached to the page.",
      "slug": "URL slug (required for standalone reports).",
    },
  },
  "create_user_tool": {
    description: "Create a user-defined tool in Galaxy from a YAML tool definition.\n\nUser-defined tools are lightweight, containerized tools that can be created\nwithout admin privileges. They are stored in the database, scoped to the\ncreating user, and can be embedded in workflows (importing the workflow\nautomatically creates the tool for the importing user).",
    parameters: {
      "representation": "The tool definition as a dictionary matching the\nGalaxyUserTool schema. Required fields:\n- class: \"GalaxyUserTool\" (exactly this string)\n- id: tool identifier (lowercase, no spaces, 3-255 chars)\n- version: version string (e.g. \"0.1.0\")\n- name: display name shown in Galaxy tool menu\n- container: container image as a STRING, NOT a dict (a common\n  mistake). Prefer a real biocontainer over a bare image -- a bare\n  image like \"python:3.12-slim\" ships no third-party libraries (it\n  cannot `import pandas`). If the recommend_biocontainer tool is\n  available it resolves a verified image for you; otherwise use a\n  known-good biocontainer tag.\n- shell_command: the command to execute, with $(inputs.name.path)\n  for data inputs and $(inputs.name) for parameter inputs\n- inputs: list of input dicts, each with \"name\" and \"type\"\n  (type can be: \"data\", \"integer\", \"float\", \"text\", \"boolean\")\n- outputs: list of output dicts, each with \"name\", \"type\": \"data\",\n  \"format\" (e.g. \"tabular\", \"vcf\", \"bed\"), and \"from_work_dir\"",
    },
  },
  "delete_user_tool": {
    description: "Deactivate a user-defined tool. Deactivated tools are not loaded into the toolbox.",
    parameters: {
      "uuid": "The UUID of the tool to deactivate. Get this from list_user_tools().",
    },
  },
  "download_dataset": {
    description: "Download a dataset from Galaxy to the local filesystem or memory",
    parameters: {
      "dataset_id": "Galaxy dataset ID - a hexadecimal hash string identifying the dataset\n       (a 16-character hex string)",
      "file_path": "Local filesystem path where to save the downloaded file\n      (e.g., '/path/to/data.txt', requires write access to filesystem)\n      If not provided, downloads to memory instead",
      "use_default_filename": "Deprecated - use file_path for specific locations\n                 (default: True, ignored when file_path not provided)",
      "require_ok_state": "Only allow download if dataset processing state is 'ok'\n             (default: True, set False to download datasets in other states)",
    },
  },
  "get_collection_details": {
    description: "Get detailed information about a dataset collection and its members\n\nDataset collections group multiple datasets together (e.g., paired-end reads,\nsample lists). This tool shows the collection structure and member datasets.",
    parameters: {
      "collection_id": "Galaxy dataset collection ID - a hexadecimal hash string\n          (e.g., 'a1b2c3d4e5f6g7h8', typically 16 characters)",
      "max_elements": "Maximum number of collection elements to return (default: 100)\n         Set lower for large collections to avoid overwhelming output",
    },
  },
  "get_dataset_details": {
    description: "Get detailed information about a specific dataset, optionally including a content preview",
    parameters: {
      "dataset_id": "Galaxy dataset ID - a hexadecimal hash string identifying the dataset\n       (a 16-character hex string)",
      "include_preview": "Whether to include a preview of the dataset content showing first N lines\n            (default: True, only works for datasets in 'ok' state)",
      "preview_lines": "Number of lines to include in the content preview (default: 10)",
    },
  },
  "get_histories": {
    description: "Get list of user's histories with optional pagination and filtering.\n\nHistories are Galaxy's primary organizational unit - each contains datasets,\ncollections, and records of analyses. Most operations require a history_id.\n\nRECOMMENDED WORKFLOW:\n1. Call get_histories() to see existing histories\n2. Either use an existing history_id or create_history() for new work\n3. Upload data or run tools in the selected history",
    parameters: {
      "limit": "Maximum histories to return. Default None returns all.\n   Use with offset for pagination on large history lists.",
      "offset": "Skip this many histories (for pagination). Default 0.",
      "name": "Return only histories with exactly this name. The match is exact and\n  case-sensitive, not a substring: name=\"RNA\" does not match\n  \"RNA-seq analysis\".",
    },
  },
  "get_history_contents": {
    description: "Get paginated contents (datasets and collections) from a specific history with ordering support",
    parameters: {
      "history_id": "Galaxy history ID - a hexadecimal hash string identifying the history\n       (a 16-character hex string)",
      "limit": "Maximum number of items to return per page (default: 100). A page too large\n   for the output budget is cut short; pagination says where the next one starts.",
      "offset": "Number of items to skip from the beginning (default: 0, for pagination)",
      "deleted": "Include deleted datasets in results (default: False)",
      "visible": "Include only visible datasets (default: True, set False to include hidden)",
      "order": "Sort order for results: hid, create_time, update_time, name, extension\n  or size, followed by '-asc' or '-dsc'. 'hid-asc' (default) is oldest\n  first.",
    },
  },
  "get_history_details": {
    description: "Get history metadata and summary count ONLY - does not return actual datasets\n\nThis function provides quick access to history information without loading all datasets.\nFor the actual datasets/contents, use get_history_contents() which supports\npagination and ordering.",
    parameters: {
      "history_id": "Galaxy history ID - a hexadecimal hash string identifying the history\n       (a 16-character hex string)",
    },
  },
  "get_invocations": {
    description: "View workflow invocations in Galaxy",
    parameters: {
      "invocation_id": "Specific workflow invocation ID to view - a hexadecimal hash string\n          (a 16-character hex string, optional)",
      "workflow_id": "Filter invocations by workflow ID - a hexadecimal hash string\n        (a 16-character hex string, optional)",
      "history_id": "Filter invocations by history ID - a hexadecimal hash string\n       (a 16-character hex string, optional)",
      "limit": "Maximum number of invocations to return. Leave it unset and none is\n   sent, so Galaxy applies its own default of 20 -- not \"no limit\". Raise\n   it to see more.",
      "view": "Level of detail to return - 'element' for detailed or 'collection' for summary\n (default: 'collection')",
      "step_details": "Include details on individual workflow steps -- each step's\n         jobs. Applies to one invocation by id, and to a listing when\n         view is 'element' (default: False)",
    },
  },
  "get_iwc_workflow_details": {
    description: "Get comprehensive details about a specific IWC workflow before importing.\n\nUse this to examine a workflow's full documentation, inputs, and complexity\nbefore deciding to import it into your Galaxy instance.\n\nRECOMMENDED WORKFLOW:\n1. Search workflows with search_iwc_workflows() or recommend_iwc_workflows()\n2. Call this function with the trsID to get full details\n3. Review the readme and inputs to ensure it fits your needs\n4. Import with import_workflow_from_iwc(trs_id)",
    parameters: {
      "trs_id": "The TRS (Tool Registry Service) ID from search results.\n    Format: \"#workflow/github.com/iwc-workflows/<name>/<branch>\"\n    Example: \"#workflow/github.com/iwc-workflows/rnaseq-pe/main\"",
    },
  },
  "get_iwc_workflows": {
    description: "List workflows published by the IWC (Intergalactic Workflow Commission).\n\nReturns one page of workflow summaries - the same shape search_iwc_workflows\nreturns - not the raw manifest entries. A single raw entry carries the whole\nworkflow definition: median ~50 KB, largest ~500 KB, so even one of them can\noverflow an MCP client's output limit. Use get_iwc_workflow_details(trs_id) for\nthe full record.",
    parameters: {
      "limit": "Maximum workflows to return per page (default 20, max 100). A page\n   is also cut short when it would not fit the output budget.",
      "offset": "Skip this many workflows (default 0). Pass pagination.next_offset\n    to walk to the following page.",
    },
  },
  "get_job_details": {
    description: "Get detailed information about the job that created a specific dataset\n\nThe job is read in full, so a failed job's logs come back with it: tool_stdout,\ntool_stderr, job_stdout, job_stderr, stdout and stderr. A log longer than 4 KB keeps its\nfirst and last 2 KB, cut on line boundaries, with a line saying how much was left out.",
    parameters: {
      "dataset_id": "Galaxy dataset ID - a hexadecimal hash string identifying the dataset\n       (a 16-character hex string)",
      "history_id": "Galaxy history ID containing the dataset - optional for performance optimization\n       (a 16-character hex string)",
    },
  },
  "get_page": {
    description: "Get a page and the content of its latest revision.\n\nReturns `content_editor`: the editable Galaxy-flavored markdown, with\nENCODED ids in directives (e.g. `history_dataset_id=<encoded-dataset-id>`).\nThis is the form to edit and pass back to update_page. Also returns\n`content_hash`, Galaxy's page hash of that source: read the page again and\ncompare it to tell whether anyone changed the page since.",
    parameters: {
      "page_id": "Encoded id of the page (from list_pages / create_page).",
      "include_rendered": "When True, also return `content` -- the\nembed-expanded render form (inlined dataset previews). Can be\nlarge; omit unless you need the rendered output.",
    },
  },
  "get_page_revision": {
    description: "Get the content of a single page revision.\n\nRequires Galaxy 26.1 or newer.\n\nEdit `content_editor` and pass it to update_page; `content` is the same\ndocument with its embeds expanded for export, not the editable form. Galaxy\nonly began sending content_editor on a revision after 26.1, so check\n`content_editor_source`: \"content\" means the server sent none and the\neditable text you are being handed is that expanded form -- editing it and\nsaving it bakes the expansion into the page.",
    parameters: {
      "page_id": "Encoded id of the page.",
      "revision_id": "Encoded id of the revision (from list_page_revisions).",
    },
  },
  "get_server_info": {
    description: "Get Galaxy server information including version, URL, and configuration details\n\nAlso reports which tools this server is too old to run, in `unsupported_tools`,\nand whether its version could be read at all, in `version_known` -- an empty\n`unsupported_tools` means nothing is ruled out only when `version_known` is true.\n\nReturns:\n    GalaxyResult with server information in data field",
    parameters: {

    },
  },
  "get_tool_citations": {
    description: "Get citation information for a specific tool",
    parameters: {
      "tool_id": "ID of the tool",
    },
  },
  "get_tool_details": {
    description: "Get detailed information about a specific tool including its input parameters.\n\nRECOMMENDED WORKFLOW:\n1. First find tools using search_tools_by_name() or get_tool_panel()\n2. Call this function with io_details=True to see all input parameters\n3. Use the inputs schema to construct the inputs dict for run_tool()",
    parameters: {
      "tool_id": "Galaxy tool identifier. Common formats:\n     - Simple: \"fastqc\", \"bwa\", \"upload1\"\n     - Toolshed: \"toolshed.g2.bx.psu.edu/repos/devteam/fastqc/fastqc/0.73\"",
      "io_details": "Set True to include detailed input/output parameter schemas.\n        Essential for understanding how to call run_tool().",
    },
  },
  "get_tool_input_template": {
    description: "Return a ready-to-fill ``inputs`` skeleton for a tool, plus a compact schema.\n\nCall this before run_tool when you are unsure how to shape ``inputs``. Replace\nplaceholders (e.g. ``<dataset_id>``) with real values. Repeats show one\ninstance (``name_0|...``); duplicate with ``name_1|...`` to add more. The\nflattened-key convention is ``section|param``, ``cond|selector``,\n``repeat_0|param``.",
    parameters: {
      "tool_id": "",
    },
  },
  "get_tool_panel": {
    description: "Browse the Galaxy tool panel (toolbox) one level at a time.\n\nThe whole panel is megabytes on a production server, so it is never returned\nwhole. Called with no arguments this lists the top-level entries - each section\nwith the number of tools in it, plus any tools that sit outside a section.\nPass section_id to list the tools in one section.",
    parameters: {
      "section_id": "Panel section to open, from a previous summary call. Omit to\n        list the sections themselves.",
      "limit": "Maximum entries to return per page (default 100, max 500). A page is\n   also cut short when it would not fit the output budget.",
      "offset": "Skip this many entries (default 0). Pass pagination.next_offset\n    to walk to the following page.",
    },
  },
  "get_tool_run_examples": {
    description: "Return the exact XML test definitions (inputs, outputs, assertions, required files)\nfor a Galaxy tool so an LLM can study real, working run configurations.",
    parameters: {
      "tool_id": "ID of the tool to inspect",
      "tool_version": "Optional version selector (use '*' for all versions)",
    },
  },
  "get_user": {
    description: "Get current user information\n\nReturns:\n    GalaxyResult with current user details in data field",
    parameters: {

    },
  },
  "get_workflow_details": {
    description: "Get detailed information about a specific workflow",
    parameters: {
      "workflow_id": "ID of the workflow to get details for - a hexadecimal hash string",
      "version": "Specific version of the workflow (optional, uses latest if not specified)",
    },
  },
  "get_workflow_input_template": {
    description: "Return a ready-to-fill template plus a run guide for a workflow.\n\nCall this before invoke_workflow. Each slot lists its label, expected source\n(hda/hdca), accepted datatypes, collection type, and -- for parameters --\nselectable `options` as [{label, value}]. On the `style=run` path these are\nGalaxy-resolved: reference-genome dbkeys come from the server regardless of\nhistory, and passing `history_id` additionally surfaces that history's\ncompatible datasets as candidates. The legacy .ga fallback carries only the\nstatic restrictions baked into the workflow -- no server- or history-resolved\nvalues -- see `guide.notes`. `guide` carries a short description and provenance.\nFill `inputs_template` (keyed by step_index) and invoke with\n`inputs_by=\"step_index|step_uuid\"`. Pass `verbose=True` for the full readme\nand uncapped option lists. `warnings` flags legacy patterns.",
    parameters: {
      "workflow_id": "",
      "history_id": "",
      "verbose": "",
    },
  },
  "import_workflow_from_iwc": {
    description: "Import a workflow from IWC to the user's Galaxy instance",
    parameters: {
      "trs_id": "TRS ID of the workflow in the IWC manifest",
    },
  },
  "invoke_workflow": {
    description: "Invoke (run) a workflow with specified inputs and parameters",
    parameters: {
      "workflow_id": "ID of the workflow to invoke - a hexadecimal hash string",
      "inputs": "Mapping of workflow inputs to datasets. Format:\n   {'step_index': {'id': 'dataset_id', 'src': 'hda'}} where src can be:\n   - 'hda' for HistoryDatasetAssociation\n   - 'hdca' for HistoryDatasetCollectionAssociation\n   - 'ldda' for LibraryDatasetDatasetAssociation\n   - 'ld' for LibraryDataset",
      "params": "Tool parameter overrides as a nested dictionary",
      "history_id": "ID of history to store workflow outputs (optional)",
      "history_name": "Name for new history to create (ignored if history_id provided)",
      "inputs_by": "How to identify workflow inputs - 'step_index', 'step_uuid', 'name', or\n      'step_index|step_uuid' (recommended; matches get_workflow_input_template)",
      "parameters_normalized": "Whether parameters are already in normalized format",
    },
  },
  "list_history_ids": {
    description: "Get a simplified, paginated list of history IDs and names for easy reference",
    parameters: {
      "limit": "Maximum histories to return per page (default 100, max 500). A page\n   is also cut short when it would not fit the output budget.",
      "offset": "Skip this many histories (default 0). Pass pagination.next_offset\n    to walk to the following page.",
    },
  },
  "list_page_revisions": {
    description: "List the revision history of a page.\n\nRequires Galaxy 26.1 or newer.\n\nEach revision carries an `edit_source` (\"user\", \"agent\", or \"restore\")\nrecording who made the edit.",
    parameters: {
      "page_id": "Encoded id of the page.",
      "sort_desc": "Newest-first when True (default oldest-first).",
    },
  },
  "list_pages": {
    description: "List Galaxy pages (markdown documents) viewable by the user.\n\nRequires Galaxy 26.1 or newer.\n\nThe history is what needs the newer server, and the whole tool is refused\nrather than only a filtered call: an older one has no history_id query\nparameter and no history_id on a page, so it would answer with every page\nthe user can see and no way to tell a notebook from a report.\n\nA page attached to a history is a \"Galaxy Notebook\"; a standalone page is\na \"Report\". Pass history_id to list only that history's notebooks.",
    parameters: {
      "history_id": "Encoded history id. When set, returns only notebooks\nattached to that history.",
      "search": "Freetext filter over title/content.",
      "limit": "Max pages to return (default 100).",
      "offset": "Pagination offset.",
      "show_published": "Also include pages published by other users (default False).",
      "show_shared": "Also include pages shared with the user (default False).",
    },
  },
  "list_user_tools": {
    description: "List user-defined tools belonging to the current user, one page at a time.",
    parameters: {
      "active": "If True (default), only show active tools. Set False to include deactivated tools.",
      "limit": "Maximum tools to return per page (default 25, max 100). Each entry\ncarries the tool's full representation, so a page is often cut short to\nfit the output budget; walk pagination.next_offset for the rest.",
      "offset": "Skip this many tools (default 0). Pass pagination.next_offset to\nwalk to the following page.",
    },
  },
  "list_workflows": {
    description: "List workflows available in the Galaxy instance, one page at a time",
    parameters: {
      "name": "Filter workflows by name (optional)",
      "published": "Include published workflows (default: False, shows only user workflows)",
      "limit": "Maximum workflows to return per page (default 50, max 200). A page\n   is also cut short when it would not fit the output budget.",
      "offset": "Skip this many workflows (default 0). Pass pagination.next_offset\n    to walk to the following page.",
    },
  },
  "recommend_biocontainer": {
    description: "Resolve a verified quay.io/biocontainers image for a set of conda packages.\n\nUse this to pick the ``container`` for create_user_tool instead of guessing an\nimage. The result is verified against quay.io rather than hallucinated, which\navoids the most common user-defined-tool failure: inventing a tag, or using a\nbare image (e.g. \"python:3.12-slim\") that doesn't ship the libraries the tool\nimports.",
    parameters: {
      "packages": "The conda packages the tool wraps, each as \"name\" or\n\"name=version\" (e.g. [\"samtools=1.17\", \"bwa\"]). Use canonical conda\nnames you would `conda install` (e.g. \"pandas\", \"r-ggplot2\",\n\"samtools\"). A single package yields a single-package image; several\nyield a mulled-v2 image.",
    },
  },
  "recommend_iwc_workflows": {
    description: "Semantic search for IWC workflows based on natural language description.\n\nUse this when you have a general analysis goal and want to find the best\nmatching workflows. Uses BM25 ranking to search across names, descriptions,\nreadmes, tags, and tool names.\n\nRECOMMENDED WORKFLOW:\n1. Describe your analysis goal in natural language\n2. Review ranked recommendations with match explanations\n3. Get details for promising workflows: get_iwc_workflow_details(trs_id)\n4. Import the best match: import_workflow_from_iwc(trs_id)",
    parameters: {
      "intent": "Natural language description of your analysis goal.\n    Examples:\n    - \"I have paired-end RNA-seq data and want differential expression\"\n    - \"Assemble a bacterial genome from nanopore reads\"\n    - \"Variant calling from whole exome sequencing data\"\n    - \"Quality control for Illumina sequencing data\"",
      "limit": "Maximum number of recommendations to return (default 5, max 25)",
    },
  },
  "revert_page_revision": {
    description: "Roll a page back to an earlier revision.\n\nRequires Galaxy 26.1 or newer.\n\nCreates a NEW revision from the target revision's content, recorded with\nedit_source=\"restore\" (the history is append-only -- nothing is deleted).\nThe new revision comes back the way get_page_revision returns one, with\n`content_editor` and the `content_editor_source` that says where it came from.",
    parameters: {
      "page_id": "Encoded id of the page.",
      "revision_id": "Encoded id of the revision to restore.",
    },
  },
  "run_tool": {
    description: "Run a Galaxy tool on datasets in a history.\n\nRECOMMENDED WORKFLOW:\n1. Create or select a history: create_history() or get_histories()\n2. Upload data: upload_file() or upload_file_from_url()\n3. Get tool parameters: get_tool_details(tool_id, io_details=True)\n4. Call this function with properly formatted inputs\n5. Monitor job: get_job_details() or check history contents",
    parameters: {
      "history_id": "Galaxy history ID (a 16-character hex string).\n        Get from create_history() or get_histories().",
      "tool_id": "Tool identifier. Common formats:\n     - Simple built-in: \"cat1\", \"Cut1\", \"upload1\"\n     - Toolshed: \"toolshed.g2.bx.psu.edu/repos/iuc/fastqc/fastqc/0.73\"",
      "inputs": "Tool input parameters. Dataset inputs use this format:\n    {\"input_name\": {\"src\": \"hda\", \"id\": \"dataset_id\"}}",
      "tool_version": "Ask Galaxy for a specific version of the tool rather than\n          whichever one it would pick. It is a request, not a guarantee:\n          Galaxy falls back to an installed version when the one asked\n          for is not there. Omit it and Galaxy chooses, which is the\n          default and almost always what you want. Versions come from\n          get_tool_details(tool_id).",
    },
  },
  "run_user_tool": {
    description: "Run a user-defined tool via the standard Galaxy tools API.\n\nSubmits the UDT with POST /api/tools carrying the tool UUID. This\nsynchronous endpoint runs user-defined tools on every Galaxy that\nsupports them and needs no Celery -- unlike the newer POST /api/jobs\ntool-request path, which is Celery-only and 500s for UDTs on 26.0.\nIt returns the job and output dataset handles immediately; the job\nthen runs asynchronously on the cluster (poll it for state).",
    parameters: {
      "history_id": "Galaxy history ID where outputs will be placed.",
      "tool_uuid": "The UUID of the user-defined tool (from create_user_tool or list_user_tools).",
      "inputs": "Tool input parameters. Dataset inputs use:\n    {\"input_name\": {\"src\": \"hda\", \"id\": \"dataset_id\"}}\n    Scalar parameters use direct values:\n    {\"param_name\": value}",
    },
  },
  "search_iwc_workflows": {
    description: "Search for workflows in the IWC (Intergalactic Workflow Commission) manifest.\n\nIWC hosts curated, best-practice workflows for common bioinformatics analyses.\nThis function searches across workflow names, descriptions, tags, and readmes.\nResults are paginated: pagination.total_items is how many workflows matched,\npagination.next_offset is where the next page starts.\n\nRECOMMENDED WORKFLOW:\n1. Search for workflows matching your analysis need\n2. Review the results - check step_count for complexity, readme_summary for details\n3. Call get_iwc_workflow_details(trs_id) for full information\n4. Import with import_workflow_from_iwc(trs_id)\n5. Run with invoke_workflow()",
    parameters: {
      "query": "Search query (case-insensitive). Matches against:\n   - Workflow name (e.g., \"RNA-seq\")\n   - Description/annotation\n   - Tags (e.g., \"assembly\", \"transcriptomics\")",
      "limit": "Maximum workflows to return per page (default 20, max 100). A broad\n   query matches most of the 123-workflow corpus, which is ~184 KB\n   unpaged and more than an MCP client will pass through intact.",
      "offset": "Skip this many matches (default 0). Pass pagination.next_offset\n    to walk to the following page.",
    },
  },
  "search_tools_by_keywords": {
    description: "Recommend Galaxy tools based on a list of keywords.\n\nMatches on name and description first, then on accepted input formats, and keeps\nthat order so paging is stable. Results are paginated: pagination.total_items is\nthe full match count, pagination.next_offset is where the next page starts.\n\nExpensive: finding the format matches costs one detail lookup per tool that did\nnot match by name, and every page repeats the whole scan. Prefer\nsearch_tools_by_name when a name or description substring will do.",
    parameters: {
      "keywords": "A list of keywords or phrases describing what you're looking for,\ne.g., [\"csv\", \"rna\", \"alignment\", \"visualization\"]. The search will match tools\nwhose name, description, or accepted input formats contain any of these keywords.",
      "limit": "Maximum tools to return per page (default 50, max 200). A page is\n   also cut short when it would not fit the output budget.",
      "offset": "Skip this many matches (default 0). Pass pagination.next_offset for\nthe following page.",
    },
  },
  "search_tools_by_name": {
    description: "Search Galaxy tools whose name, ID, or description contains the given query (substring match).\n\nResults are paginated. pagination.total_items is how many tools matched in total;\npagination.next_offset is where the next page starts (None on the last page).\n\nRECOMMENDED WORKFLOW:\n1. Use this function to find tools by name/keyword\n2. Review the returned tool IDs and names\n3. Call get_tool_details(tool_id) for full input parameters\n4. Call run_tool() with the correct inputs",
    parameters: {
      "query": "Search query - matches against tool name, ID, or description.\n   Examples: \"fastq\", \"alignment\", \"filter\", \"bwa\"",
      "limit": "Maximum tools to return per page (default 25, max 100). A page is\n   also cut short when it would not fit the output budget, so ask for\n   what you want and walk pagination.next_offset.",
      "offset": "Skip this many matches (default 0). Pass pagination.next_offset\n    to walk to the following page.",
    },
  },
  "update_history": {
    description: "Update an existing Galaxy history's metadata.\n\nAny combination of name, annotation, tags, deleted, and published can be updated\nin a single call. Fields left as None are not modified.",
    parameters: {
      "history_id": "The ID of the history to update. Obtain this from get_histories()\n        or list_history_ids().",
      "name": "New name for the history (optional).",
      "annotation": "New annotation/description text for the history (optional).",
      "tags": "New list of tags for the history (optional). Replaces any existing tags.",
      "deleted": "If True, soft-delete the history; if False, restore a deleted history\n     (optional).",
      "published": "If True, publish the history (make it public); if False, unpublish\n       (optional).",
    },
  },
  "update_page": {
    description: "Update a page, creating a new revision when content changes.\n\nRequires Galaxy 26.1 or newer.\n\nReplace one section instead of the whole page by passing section_heading\n(the exact heading line, e.g. '## Methods') with section_content (its new\ntext, heading line included); a heading the page lacks is appended, and every\nsection with that heading line is replaced, as Galaxy's page editor does.\nMarkdown pages only. Pass the\ncontent_hash get_page returned as expect_hash and the write is refused if the\npage changed since you read it.\n\nContent is Galaxy-flavored markdown using ENCODED ids in directives\n(e.g. `history_dataset_id=<encoded-dataset-id>`) -- never raw integer ids or\nHIDs. Get encoded ids from get_history_contents / get_dataset_details.\nEdits made through this tool are recorded with edit_source=\"agent\". This does\nnot change the page's content_format, so editing a page originally authored\nas HTML with markdown content can mislabel it.",
    parameters: {
      "page_id": "Encoded id of the page.",
      "content": "New markdown content. Omit to leave content unchanged.",
      "title": "New title. Omit to leave unchanged.",
      "section_heading": "The exact heading line of the section to replace (markdown\npages only; every section with that heading line is replaced).",
      "section_content": "That section's new text, heading line included.",
      "expect_hash": "content_hash from when the page was read; refused if it changed.",
    },
  },
  "upload_file": {
    description: "Upload a local file to Galaxy for analysis.\n\nGalaxy automatically detects the file type (FASTQ, BAM, BED, etc.) and\nindexes the file appropriately. Large files are uploaded efficiently.\n\nRECOMMENDED WORKFLOW:\n1. Create a history: create_history(\"My Analysis\")\n2. Upload your data files with this function\n3. Wait for upload to complete (check dataset state)\n4. Run tools on the uploaded data: run_tool()",
    parameters: {
      "path": "Local file path to upload. Supports common bioinformatics formats:\n  - Sequences: .fastq, .fasta, .fa, .fq, .fastq.gz\n  - Alignments: .bam, .sam, .cram\n  - Annotations: .bed, .gff, .gtf, .vcf\n  - Tabular: .csv, .tsv, .txt",
      "history_id": "Target history ID. If None, uses the most recent history.\n        Recommend always specifying for clarity.",
    },
  },
  "upload_file_from_url": {
    description: "Upload a file from a URL to Galaxy",
    parameters: {
      "url": "URL of the file to upload (e.g., 'https://example.com/data.fasta')",
      "history_id": "Galaxy history ID where to upload the file - optional, uses current history\n       (a 16-character hex string)",
      "file_type": "Galaxy file format name (default: 'auto' for auto-detection)\n      Common types: 'fasta', 'fastq', 'bam', 'vcf', 'bed', 'tabular', etc.",
      "dbkey": "Database key/genome build (default: '?', e.g., 'hg38', 'mm10', 'dm6')",
      "file_name": "Optional name for the uploaded file in Galaxy (inferred from URL if not provided)",
    },
  },
};
