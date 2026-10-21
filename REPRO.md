# How the measurements were made (outline)

Version 0.1 contains the report and the logs. The engine program, its source and the build scripts are not included yet; this file outlines the steps so that the logs can be read in context.

1. **Weights.** Download DeepSeek-V4.1-Flash-1017 (510.3 GB) from DeepSeek's release and verify the files.
2. **Header check and non-expert model file.** Check the safetensors headers, index every routed expert and convert the non-expert weights (attention, shared experts, embeddings, norms) as released.
3. **2-bit copies and correction planes.** Write a 2-bit copy of each of the 15,360 routed experts, and a correction plane that turns each copy back into the exact FP4 expert.
4. **FP4 copies and mirrors.** Copy the first layers' FP4 experts to both drives so that a read can use the less busy drive.
5. **Draft head.** Write the draft head's experts in the same 2-bit format (engine only).
6. **Run.** Generate 160 tokens twice in one process for each prompt (code, reasoning, agent), with the settings in Appendix A of the report. Quality audits run with the correction gate at 0.9 and count slips for any lower gate.

Disk: about 1,009 GiB in all (Appendix B of the report). The logs in `logs/` are the unedited output of these runs; RESULTS.md says which folder holds which table.
