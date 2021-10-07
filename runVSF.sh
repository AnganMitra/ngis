inputDir=./BKDataCleaned/
start_index=0
end_index=10000
groupby=zone #domain, 
expMode=456f
taskType=vsfGen #optVsF, zonAly
mkdir -p paperAnalysis/
python3 geneticOptimizer/main.py -i $inputDir -ts $start_index -te $end_index -g $groupby -c $expMode -tk $taskType