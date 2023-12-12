# VecchiaNN

VecchiaNN is an R package for calculating the ordered nearest neighbors as required by Vecchia Approximation. The software is designed to run on an NVIDIA GPU. Refer to the vignette for usage. Installation instructions are below.

### Installation

We recommend installing on G2. Information on G2 can be found at https://it.coecis.cornell.edu/researchit/g2cluster/. You may have to submit a help ticket to set up a G2 account. You will also have to install the Cornell VPN https://it.cornell.edu/cuvpn.

ssh into G2. Create a virtual environment with Anaconda that includes R.

```Shell
netid@g2-login-01:~$ /share/apps/anaconda3/2021.05/bin/conda init
netid@g2-login-01:~$ conda create -n r-env r-essentials r-base
netid@g2-login-01:~$ conda activate r-env
```

Clone GpGpU and install the package.

```Shell
(r-env) netid@g2-login-01:~$ git clone https://github.com/zjames12/VecchiaNN.git
(r-env) netid@g2-login-01:~$ R CMD build VecchiaNN
(r-env) netid@g2-login-01:~$ R CMD INSTALL VecchiaNN_1.0.tar.gz
```

If you are reviewing this package for stat computing please contact me if you need help.

### Usage

Here is an example script that will generate a lattice data set and calculate the ordered nearest neighbors matrix. Create an R script on the G2 server with the below code. We will name the file `nn.R`.

```R
library(VecchiaNN)

# Create the data
n1 = 30
n2 = 30
locs <- as.matrix( expand.grid( (1:n1), (1:n2) ) )

# Calculate the nearest neighbors
m = 10
NNarray <- VecchiaNN::nearest_neighbors(locs, m, precision = "single")
NNarray
```

Create a submission script with the follow code. Replace `netid` with your Cornell NetID. We will name the file `test.sub`.

```Shell
#!/bin/bash
#SBATCH -J nn                         # Job name
#SBATCH -o nn_results.out                  # output file
#SBATCH -e nn_results.err                  # error log file
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 1                                 # Total number of cores requested
#SBATCH --cpus-per-task=1
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=86384                           # server memory requested (per node)
#SBATCH -t 1:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=default_partition        # Request partition
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
Rscript /home/netid/nn.R
```
Submit the script.

```Shell
(r-env) netid@g2-login-01:~$ sbatch --requeue test.sub
```

The status of the code can be checked with `squeue -l`. Once execution has completed the result can be viewed in the output file.

```Shell
(r-env) netid@g2-login-01:~$ cat nn_results.out
```
