# VecchiaNN

VecchiaNN is an R package for calculating the ordered nearest neighbors as required by Vecchia Approximation. The software is designed to run on an NVIDIA GPU. Refer to the vignette for usage. Installation instructions are below.

### Installation

We recommend installing on G2. Information on G2 can be found at https://it.coecis.cornell.edu/researchit/g2cluster/. You may have to submit a help ticket to set up a G2 account. You will also have to install the Cornell VPN https://it.cornell.edu/cuvpn.

ssh into G2. Create a virtual environment with Anaconda that includes R.

```Shell
netid@g2-login-01:~$ conda create -n r-env r-essentials r-base
netid@g2-login-01:~$ conda activate r-env
```

Clone GpGpU and install the package.

```Shell
netid@g2-login-01:~$ git clone https://github.com/zjames12/VecchiaNN.git
netid@g2-login-01:~$ R CMD build VecchiaNN
netid@g2-login-01:~$ R CMD INSTALL VecchiaNN_1.0.tar.gz
```

If you are reviewing this package for stat computing please contact me if you need help.

### Usage

Here is an example script that will generate a lattice data set and calculate the ordered nearest neighbors matrix

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
