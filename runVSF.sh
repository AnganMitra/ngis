inputDir=./BKDataCleaned/
outputDir=./paperAnalysis/
start_index=0
end_index=10000
# groupby=zone 
# groupby=domain
# expMode=4567f 
# expMode=67f
# expMode=456f
# expMode=56f
# expMode=47f
taskType=vsfGen 
# taskType=optVsF
mkdir -p paperAnalysis/
# python3 geneticOptimizer/main.py -i $inputDir -ts $start_index -te $end_index -g $groupby -c $expMode -tk $taskType -o $outputDir
# python3 geneticOptimizer/predictionAnalysis.py

expMode=( 4567f 67f 456f 56f 47f)
groubyMode=(domain zone)
for expInput in "${expMode[@]}"
do
    for groupType in "${groubyMode[@]}"
    do
        python3 geneticOptimizer/main.py -i $inputDir -ts $start_index -te $end_index -g $groupType -c $expInput -tk $taskType -o $outputDir
    done
done