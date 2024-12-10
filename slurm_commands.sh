module load git/2.19.0 python/3.9.16 cuda/12.4
source /projects/illinois/class/cs444/saurabhg/fa2024/mp3/venv/bin/activate
sbatch --export=ALL,OUTPUT_DIR="runs/run1/" --output="runs/run1/%j.out" --error="runs/run1/%j.err" train.sbatch
sbatch --export=ALL,OUTPUT_DIR="runs/test/" --output="runs/test/%j.out" --error="runs/test/%j.err" train.sbatch
sbatch --export=ALL,OUTPUT_DIR="runs/run3/" --output="runs/run3/%j.out" --error="runs/run3/%j.err" demo.sbatch
sbatch --export=ALL,OUTPUT_DIR="runs/effi/" --output="runs/effi/%j.out" --error="runs/effi/%j.err" effi.sbatch
sbatch --export=ALL,OUTPUT_DIR="runs/getnum/" --output="runs/getnum/%j.out" --error="runs/getnum/%j.err" getnum.sbatch
scancel "jobid"

tail -f runs/test/log.txt

squeue -u muchenx2


while true; do
    squeue -u muchenx2
    sleep 10
done


sbatch --export=ALL,OUTPUT_DIR="runs/exp/" --output="runs/exp/%j.out" --error="runs/exp/%j.err" train.sbatch
sbatch --export=ALL,OUTPUT_DIR="runs/exp_2/",NUM_RESIDUAL_BLOCKS="2" --output="runs/exp_2/%j.out" --error="runs/exp_2/%j.err" train.sbatch
sbatch --export=ALL,OUTPUT_DIR="runs/exp_3/",NUM_RESIDUAL_BLOCKS="3" --output="runs/exp_3/%j.out" --error="runs/exp_3/%j.err" train.sbatch
sbatch --export=ALL,OUTPUT_DIR="runs/exp_f/",USE_BN="0" --output="runs/exp_f/%j.out" --error="runs/exp_f/%j.err" train.sbatch
sbatch --export=ALL,OUTPUT_DIR="runs/exp_leakyrelu/",NONLINEARITY="leakyrelu" --output="runs/exp_leakyrelu/%j.out" --error="runs/exp_leakyrelu/%j.err" train.sbatch
sbatch --export=ALL,OUTPUT_DIR="runs/exp_gelu/",NONLINEARITY="gelu" --output="runs/exp_gelu/%j.out" --error="runs/exp_gelu/%j.err" train.sbatch
sbatch --export=ALL,OUTPUT_DIR="runs/exp_relu6/",NONLINEARITY="relu6" --output="runs/exp_relu6/%j.out" --error="runs/exp_relu6/%j.err" train.sbatch

