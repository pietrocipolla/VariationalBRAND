#!/usr/bin/env Rscript
source("/home/eb/Desktop/MCMC_VI_R/BASH/pico_brand.R")

# SETUP TEST
#PREVENT CRASH FOR OUT OF MEMORY ERRORS (LINUX-UBUNTU/MINT)
#https://stackoverflow.com/questions/25000496/python-script-terminated-by-sigkill-rather-than-throwing-memoryerror
#sudo nano /etc/sysctl.conf
#add:
##vm.overcommit_memory = 2
##vm.overcommit_ratio = 100

#INPUT FROM BASH SCRIPT
# remove if using not using bash 
# and update WORKING_DIRECTORY, SEED, N_FACTOR, p
args = commandArgs(trailingOnly=TRUE)

WORKING_DIRECTORY = args[1] 
SEED = strtoi(args[2])
N_FACTOR = as.double(args[3])
p = strtoi(args[4])

#RUN TEST
print("SEED")
print(SEED)
print("N_FACTOR")
print(N_FACTOR)
print("p")
print(p)

setwd(WORKING_DIRECTORY)
set.seed(SEED)
output_df = generate_data(N_FACTOR, p)
gc()
pico_brand(output_df$X,output_df$Y,output_df$p,output_df$cl_train_label_noise, output_df$G, output_df$cl_tot)
