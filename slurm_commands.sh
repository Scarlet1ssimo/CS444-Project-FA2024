module load git/2.19.0 python/3.9.16 cuda/12.4
source /projects/illinois/class/cs444/saurabhg/fa2024/mp3/venv/bin/activate
sbatch --export=ALL,OUTPUT_DIR="runs/run1/" --output="runs/run1/%j.out" --error="runs/run1/%j.err" train.sbatch
sbatch --export=ALL,OUTPUT_DIR="runs/run2/" --output="runs/run2/%j.out" --error="runs/run2/%j.err" train.sbatch
sbatch --export=ALL,OUTPUT_DIR="runs/run3/" --output="runs/run3/%j.out" --error="runs/run3/%j.err" demo.sbatch
scancel "jobid"

tail -f runs/run1/log.txt

squeue -u muchenx2
