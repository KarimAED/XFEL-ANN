#PBS -lwalltime=10:00:00
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000

module load anaconda3/personal
module load cuda

python $HOME/XFEL-ANN/model_fit.py NewDelays --shape 20 20 20 20 20 -vv -a relu -l mae -e 40000 --batch_norm False -r 0.003 -reg l2 --drop_out 0.025
mkdir $WORK/delay_$PBS_JOBID
cp -r * $WORK/delay_$PBS_JOBID