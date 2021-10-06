inputDir=./BKDataCleaned/
start_index=0
end_index=100
groupby=zone #random, domain
expMode=456f
taskType=vsfGen
python3 geneticOptimizer/main.py -i $inputDir -ts $start_index -te $end_index -g $groupby -c $expMode -tk $taskType