#PBS -lwalltime=10:00:00
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000

module load anaconda3/personal
module load cuda

python $HOME/XFEL-ANN/delay_est.py $HOME
mkdir $WORK/delay_$PBS_JOBID
cp -r * $WORK/delay_$PBS_JOBID