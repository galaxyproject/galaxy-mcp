# Galaxy MCP Server Usage Examples

This document provides common usage patterns and examples for the Galaxy MCP server.

## Quick Start

### 1. Connect to Galaxy

**User:** "I want to analyze some genomics data. Can you help me connect to my Galaxy server?"

**AI Assistant:** Absolutely! I can help you connect to Galaxy. I'll need your Galaxy server URL and API key.
*Uses `connect()` to establish connection to the specified Galaxy instance*

"Great! I've successfully connected to your Galaxy instance at https://your-galaxy-instance.org.

Let me get some basic information about your server to make sure everything is working properly."
*Uses `get_server_info()` to retrieve server details*

"Perfect! Your Galaxy server is running version 23.1 and has the following features available:
- User registration: Enabled
- File upload via FTP: Available at ftp.galaxy.org
- Support resources: https://help.galaxyproject.org/

You're all set to start your analysis! What would you like to do first?"

### 2. Working with Histories

**User:** "What histories do I have in my account?"

**AI Assistant:** Let me check your histories.
*Uses `get_histories()` to retrieve a list of all user histories.*

"You have 5 histories. Here are the most recent three:
- RNA-seq Analysis - 2023-10-27
- Unnamed history
- ChIP-seq inputs

Would you like to see the contents of any of these?"

**User:** "Yeah, show me what's in the 'RNA-seq Analysis' one."

**AI Assistant:** Okay, I'll get the details for that history.
*Uses `get_history_details()` with the correct history ID.*

"The 'RNA-seq Analysis - 2023-10-27' history contains 8 datasets:
- `sample1_R1.fastq.gz` (FASTQ file)
- `sample1_R2.fastq.gz` (FASTQ file)
- `FastQC on sample1_R1` (HTML report)
- `FastQC on sample1_R2` (HTML report)
- `HISAT2 on sample1` (BAM alignment file)
- `featureCounts on sample1` (Counts file)
- `DESeq2 results` (Tabular file)
- `Volcano plot` (Image)

You can ask me to download any of these files, or use them as inputs for a new analysis."

### 3. Tool Discovery and Execution

**User:** "I need to check the quality of a FASTQ file I have on my computer. How can I do that?"

**AI Assistant:** I can help with that. The best tool for checking FASTQ quality is `FastQC`. I'll walk you through the process.

First, I need you to upload your FASTQ file.

*Uses `upload_file()` after the user provides the file.*

"Okay, I've uploaded your file `my_reads.fastq.gz` to a new history.

Now, I will run FastQC on it."

*Uses `search_tools('fastqc')` to find the tool ID, then `run_tool()` with the uploaded file as input.*

"I've started the FastQC analysis. This should only take a minute. The job is running, and the results will appear in your current history.

Is there anything else you'd like to do?"

### 4. Workflow Operations

**User:** "I need a workflow for ChIP-seq analysis. Can you find one for me from the public workflow registries?"

**AI Assistant:** Of course. I'll search the Interactive Workflow Composer (IWC) for ChIP-seq workflows.
*Uses `search_iwc_workflows('ChIP-seq')` to find relevant workflows.*

"I found a popular workflow called 'ChIP-seq analysis for paired-end data'. It seems to be a good match.

Would you like me to import it into your Galaxy account so you can use it?"

**User:** "Yes, please import it."

**AI Assistant:** Okay, I am importing the workflow now.
*Uses `import_workflow_from_iwc()` with the appropriate workflow ID.*

"The workflow has been successfully imported into your account. You can now see it in your list of available workflows and run it on your data."

#### Workflow Discovery and Management

**User:** "What workflows are available in my Galaxy instance?"

**AI Assistant:** I'll check your available workflows.
*Uses `list_workflows()` to retrieve user's workflows*

"You have 12 workflows available. Here are some highlights:
- RNA-seq Analysis Pipeline v2.1
- ChIP-seq Peak Calling Workflow
- Genome Assembly with Quality Control

Would you like details about any specific workflow?"

**User:** "Tell me about the RNA-seq workflow"

**AI Assistant:** Let me get the details of that workflow for you.
*Uses `get_workflow_details()` with the RNA-seq workflow ID*

"The RNA-seq Analysis Pipeline v2.1 has 8 steps:
1. Input: Raw FASTQ files (paired-end reads)
2. Quality control with FastQC
3. Trimming with Trim Galore
4. Alignment with HISAT2
5. Quantification with featureCounts
6. Differential expression with DESeq2
7. Visualization plots
8. Summary report

This workflow requires paired-end RNA-seq FASTQ files as input. Would you like to run this workflow?"

#### Running Workflows

**User:** "I want to run the RNA-seq workflow on my uploaded FASTQ files"

**AI Assistant:** I'll help you run that workflow. Let me first check your uploaded files and set up the analysis.
*Uses `get_history_contents()` to find FASTQ files, then `invoke_workflow()` with appropriate inputs*

"I found your paired FASTQ files:
- sample1_R1.fastq.gz
- sample1_R2.fastq.gz

I'm now starting the RNA-seq workflow with these inputs. The workflow has been submitted with invocation ID: inv_abc123.

The workflow is currently in 'scheduled' state. I'll monitor its progress for you."

**User:** "How is my workflow doing?"

**AI Assistant:** Let me check the status of your RNA-seq workflow.
*Uses `get_invocations()` to check workflow progress*

"Your workflow (invocation inv_abc123) is currently 'running':
- Steps 1-3: Completed ✓
- Step 4: Currently running (HISAT2 alignment)
- Steps 5-8: Waiting

The workflow is progressing normally. The alignment step typically takes 15-30 minutes depending on data size."

#### Workflow Management and Troubleshooting

**User:** "Something seems wrong with my workflow, it's been running for hours. Can you cancel it?"

**AI Assistant:** Let me check the workflow status and cancel it if needed.
*Uses `get_invocations()` to check status, then `cancel_workflow_invocation()` if appropriate*

"I can see your workflow has been stuck in the alignment step for 3 hours, which is unusual. I've cancelled the workflow invocation (inv_abc123).

The issue might be:
- Input files are very large
- Server resources are limited
- There may be a parameter issue

Would you like me to help you restart the workflow with adjusted parameters, or would you prefer to check the input data first?"

#### Complete Analysis Workflow

**User:** "I have some new RNA-seq data. Can you help me run a complete analysis from start to finish?"

**AI Assistant:** Absolutely! I'll guide you through a complete RNA-seq analysis workflow. Let me help you set this up step by step.

*Uses multiple MCP tools: `create_history()`, `upload_file()`, `search_iwc_workflows()`, `get_workflow_details()`, `invoke_workflow()`, `get_invocations()`*

"Here's what I'll do:
1. ✓ Created new history: 'RNA-seq Analysis - [Date]'
2. ✓ Uploaded your FASTQ files (2 files detected)
3. ✓ Found optimal workflow: 'RNA-seq differential expression' from IWC
4. ✓ Started workflow execution (invocation: inv_def456)

Your analysis is now running! The workflow includes:
- Quality control and trimming
- Alignment to reference genome
- Gene expression quantification
- Differential expression analysis
- Result visualization

I'll notify you when each major step completes. The full analysis typically takes 1-2 hours."

## General Advice and Troubleshooting

Here are some general tips for building a robust agent using this MCP server:

- **Always Connect First:** Ensure the `connect()` tool is called successfully before any other tools are used.
- **Use IDs Correctly:** Many tools require an ID for a history, dataset, or workflow. Make sure to extract and use the specific ID string from the objects returned by other tools.
- **Handle Errors Gracefully:** The tools will return error messages for common problems like invalid IDs, missing data, or connection issues. Your agent should be designed to catch these errors and report them to the user in a helpful way.
- **Check Server Logs:** If you encounter unexpected issues, the MCP server logs are the best place to look for detailed error messages and clues about what went wrong.
