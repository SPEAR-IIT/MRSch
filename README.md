If you use MRSch, Please cite the following paper

B. Li, et al., “MRSch: Multi-Resource Scheduling for HPC”, IEEE Cluster, 2022.

Burst buffer capacity can be set in data file.
## Training
Modify the parameter in cqsim_jsspp/Config/config_sys.set file:
```

is_training=1
```
Run:
```
cd src
python cqsim.py 
```
## Testing
Modify cqsim_jsspp/Config/config_sys.set file:
```
is_training=0
```

Run:
```
python cqsim.py 
```

