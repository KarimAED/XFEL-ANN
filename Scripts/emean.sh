#PBS -lwalltime=10:00:00
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000

module load anaconda3/personal
module load cuda

python $HOME/XFEL-ANN/model_fit.py EMean --shape 10 10 10 10 10 10 -vv -a relu -l mae -e 20000 --batch_norm False -p 1000 -reg l2
mkdir $WORK/emean_$PBS_JOBID
cp -r * $WORK/emean_$PBS_JOBID