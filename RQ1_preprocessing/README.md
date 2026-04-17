# RQ1 Preprocessing

This folder contains the initial preprocessing scripts, summary outputs, and supporting code for **RQ1** of my PhD research on low-power ecoacoustic IoT monitoring.

## Research Context

This work is part of a broader research project on **resource-aware ecoacoustic monitoring** for bird and environmental sound detection in low-power Internet of Things (IoT) systems. The main goal is to investigate how a raw-audio pre-detection pipeline can improve the practicality of on-device acoustic monitoring under energy, memory, and communication constraints.

RQ1 focuses on the **pre-detection pipeline**, especially the early processing steps that prepare incoming audio before downstream detection or classification.

## Purpose of This Folder

The purpose of this folder is to organise the code and outputs related to the early preprocessing stage of the RQ1 workflow. This includes:

- reading and inspecting audio inputs
- preparing audio windows for analysis
- applying waveform-domain preprocessing
- testing basic screening and routing logic
- generating summary outputs for inspection and comparison

This folder represents an early implementation and experimental workspace rather than a final production-ready software package.

## Current Files

### `rq1_monitor.py`
This script is used to monitor or run the current preprocessing workflow for RQ1. It may include the loading of audio data, extraction of signal measures, window-level checks, and reporting of results for analysis.

### `test.py`
This file is used for testing preprocessing functions, thresholds, or intermediate workflow steps during development.

### `test2.py`
This file contains additional experimental or alternative test code related to the preprocessing pipeline.

### `summary.csv`
This file stores summary results generated from the preprocessing experiments. It can be used to review outputs, compare conditions, or inspect how the current rules behave across tested audio segments.

## Methodological Focus

The current RQ1 preprocessing workflow is centred on a **raw-audio or waveform-domain pipeline**. The broader idea is to examine whether simple and computationally practical front-end processing can support reliable downstream detection in resource-constrained ecoacoustic devices.

The preprocessing work currently relates to the following stages:

1. **Audio loading and windowing**  
   Input recordings are divided into fixed-duration windows for analysis.

2. **Signal preparation**  
   Waveform-level preparation steps such as normalisation may be applied to reduce level variation between windows while preserving the signal structure.

3. **Screening and routing**  
   Simple decision rules may be applied to identify:
   - windows that should be dropped
   - windows that can pass directly
   - windows that may require additional enhancement

4. **Summary reporting**  
   Results from the preprocessing stage are recorded for later comparison and evaluation.

## Research Aim

The broader aim of this RQ1 work is to evaluate whether a minimal preprocessing front-end can support ecoacoustic event detection while remaining suitable for low-power embedded or IoT-based deployment.

This includes balancing:
- detection usefulness
- computational simplicity
- energy efficiency
- practical field deployment constraints

## Status

This folder currently contains **initial development files** for the RQ1 preprocessing workflow. The scripts and outputs may continue to change as the methodology is refined, thresholds are adjusted, and the experimental design progresses.

## Notes

- This folder is intended for research and experimentation.
- The code may contain prototype logic, test scripts, and intermediate outputs.
- Further refinement, restructuring, and documentation may be added as the RQ1 methodology develops.

## Author

**Botir Karimov**  
PhD Researcher  
University of Tasmania

## Project Scope

This work contributes to a larger research effort on:

**Low-power IoT ecoacoustic monitoring for avian and environmental sound detection**

with an emphasis on:
- raw-audio preprocessing
- resource-aware system design
- edge-based acoustic monitoring
- practical deployment under low-bandwidth and low-power constraints
