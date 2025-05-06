#!/bin/bash

sbatch mbpp.sbatch
sbatch mbpp_instruct.sbatch
sbatch humaneval.sbatch
sbatch humaneval_instruct.sbatch
sbatch codexglue_java.sbatch
sbatch codexglue_go.sbatch
