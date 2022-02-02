setwd('/home/eb/PycharmProjects/VariationalBRAND')

filenames = list.files(pattern="*output_csv")

data <- data.frame(matrix(ncol = 7, nrow = length(filenames)))
colnames(data) <- c("seed","n",	"p",	"n_iter",	"main_time_secs",	"main_time_mins",	"ARI")

for (i in 1:length(filenames)){ 
  data[i,] = t(read.delim(file=filenames[i], header=FALSE, sep="\n"))
}

print(data)

#VAR_BRAND
#n<5, 10 seeds, all p
seeds_list= 1:10
n_factor_list=c(0.5, 1, 2.5,5,10)*1000
p_list=c(2, 3, 5, 7, 10)

attach(data)

mean_output <- data.frame(matrix(ncol = 7, nrow = length(n_factor_list)*length(p_list)))
sd_output <- data.frame(matrix(ncol = 7, nrow = length(n_factor_list)*length(p_list)))

colnames(mean_output) <- c("num_simulations","n",	"p",	"n_iter_mean",	"main_time_secs_mean",	"main_time_mins_mean",	"ARI_mean")
colnames(sd_output) <- c("num_simulations","n",	"p",	"n_iter_sd",	"main_time_secs_sd",	"main_time_mins_sd",	"ARI_sd")

library(matrixStats)

i=1
for(nx in n_factor_list) {
  for(px in p_list) {
    selected_rows = which(n==nx & p==px)
    if(length(selected_rows)>0){
      mean_output[i,] <- c(length(selected_rows),nx,px,colMeans(data[selected_rows,4:7]))
      sd_output[i,] <- c(length(selected_rows),nx,px,colSds(as.matrix(data[selected_rows,4:7])))
    }else{
      mean_output[i,] <- c(length(selected_rows),nx,px,t(rep(-1,4)))
      sd_output[i,] <- c(length(selected_rows),nx,px,t(rep(-1,4)))
    }
    selected_rows=NULL
    i=i+1
  }
}

setwd('/home/eb/Desktop/MCMC_VI_R/BASH/tests')
write.csv(mean_output,"var_brand_output_avg_mean.csv", row.names = FALSE)
write.csv(sd_output,"var_brand_output_avg_sd.csv", row.names = FALSE)

# n_factor = 10
# RESOURCE_EXHAUSTED: Out of memory allocating 800000000 bytes.
# 
# 
# RESOURCE_EXHAUSTED: Out of memory allocating 800000000 bytes.
# 
# 
# RESOURCE_EXHAUSTED: Out of memory allocating 800000000 bytes.
# 
# 
# RESOURCE_EXHAUSTED: Out of memory allocating 1200000000 bytes.
# 
# 
# RESOURCE_EXHAUSTED: Out of memory allocating 1200000000 bytes.
# 
# 
# RESOURCE_EXHAUSTED: Out of memory allocating 1200000000 bytes.
# 
# 
# RESOURCE_EXHAUSTED: Out of memory allocating 2000000000 bytes.
# 
# 
# RESOURCE_EXHAUSTED: Out of memory allocating 2000000000 bytes.
# 
# 
# RESOURCE_EXHAUSTED: Out of memory allocating 2000000000 bytes.


#MCMC
setwd('/home/eb/Desktop/MCMC_VI_R/BASH/tests/02-02-2022_09-50-28')

filenames = list.files(pattern="output_csv*")

data <- data.frame(matrix(ncol = 7, nrow = length(filenames)))
colnames(data) <- c("seed","p",	 "n",	"elapsed_time_mins",	"tot_clusters",	
                    "cluster_found",	"ARI")

for (i in 1:length(filenames)){ 
  data[i,] = t(read.delim(file=filenames[i], header=FALSE, sep="\n"))
}

print(data)

#mcmc: n<3, 10 seeds, all <3
seeds_list= 1:10
n_factor_list=c(0.5, 1, 2.5,5,10)*1000
p_list=c(2, 3, 5, 7, 10)

attach(data)

mean_output <- data.frame(matrix(ncol = 7, nrow = length(n_factor_list)*length(p_list)))
sd_output <- data.frame(matrix(ncol = 7, nrow = length(n_factor_list)*length(p_list)))

colnames(mean_output) <- c("num_simulations","p",	 "n",	"elapsed_time_mins_mean",	"tot_clusters_mean",	"cluster_found_mean",	"ARI_mean")
colnames(sd_output) <- c("num_simulations","p",	 "n",	"elapsed_time_mins_sd",	"tot_clusters_sd",	"cluster_found_sd",	"ARI_sd")


library(matrixStats)

i=1
for(nx in n_factor_list) {
  for(px in p_list) {
    selected_rows = which(n==nx & p==px)
    if(length(selected_rows)>0){
      mean_output[i,] <- c(length(selected_rows),colMeans(data[selected_rows,-1]))
      sd_output[i,] <- c(length(selected_rows),px,nx,colSds(as.matrix(data[selected_rows,4:7])))
    }else{
      mean_output[i,] <- c(length(selected_rows),px,nx,t(rep(-1,4)))
      sd_output[i,] <- c(length(selected_rows),px,nx,t(rep(-1,4)))
    }
    selected_rows=NULL
    i=i+1
  }
}

setwd('/home/eb/Desktop/MCMC_VI_R/BASH/tests')
write.csv(mean_output,"mcmc_output_avg_mean.csv", row.names = FALSE)
write.csv(sd_output,"mcmc_output_avg_sd.csv", row.names = FALSE)


