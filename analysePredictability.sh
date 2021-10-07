outputDir=./paperAnalysis/
expMode=( 4567f 67f 456f 56f 47f)
groubyMode=(domain zone)
for expInput in "${expMode[@]}"
do
    for groupType in "${groubyMode[@]}"
    do
        python3 geneticOptimizer/predictionAnalysis.py -o $outputDir  -g $groupType -c $expInput 
    done
done