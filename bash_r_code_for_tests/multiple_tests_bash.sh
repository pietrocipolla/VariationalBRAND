#!/bin/bash
### MCMC 4H + MINI 3H IN 7 ORE FINISCE, ULTIMO SEED RICHIDERA 9 ORE. ##
echo "Hello World"
DATE_WITH_TIME=$(date +%d-%m-%Y_%H-%M-%S)
WORKING_DIRECTORY="/home/eb/Desktop/MCMC_VI_R/BASH/tests/"$DATE_WITH_TIME"/"
mkdir -p $WORKING_DIRECTORY;

#mini
#seeds_list=(1 2 3)
#n_factor_list=(5) #N FATTORE = 10 CI METTE 3 ORE*3, N = 5 45 MIN *3 => 3H +; MCMC 25MIN*10 = 4H
#p_list=(5 7 10) #2 GIA FATTO P = 3 MANCA SOLO IL SEED 9 E 10 QUINDI OK ANCHE QUELLI FINO A 3


#default
seeds_list=(1 2 3 4 5 6 7 8 9 10)
n_factor_list=(0.5 1 2.5 5 10)
p_list=(2 3 5 7 10)


VAR_BRAND
generate data
for SEED in ${seeds_list[@]}
do
    for N_FACTOR in ${n_factor_list[@]}
    do 
        for P in ${p_list[@]}
        do
           echo $SEED , $N_FACTOR , $P
           Rscript --vanilla generate_and_save_data_bash.R $WORKING_DIRECTORY $SEED $N_FACTOR $P
        done
    done
done

run multiple_var_brand
cd /home/eb/PycharmProjects/VariationalBRAND
VAR_BRAND_SEED=666
python3 root/multiple_test_main_bash.py $WORKING_DIRECTORY $VAR_BRAND_SEED


##MCMC
##regenerate data and run MCMC
#mini
seeds_list=(1 2 3 4 5 6 7 8 9 10)
n_factor_list=(0.5 1)
p_list=(2 3)

for SEED in ${seeds_list[@]}
do
    for N_FACTOR in ${n_factor_list[@]}
    do 
        for P in ${p_list[@]}
        do
           echo $SEED , $N_FACTOR , $P
           Rscript --vanilla test_mcmc_bash.R $WORKING_DIRECTORY $SEED $N_FACTOR $P
        done
    done
done

