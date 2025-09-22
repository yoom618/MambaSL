# UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket")
# UEA_MTSC30=("DuckDuckGeese" "EigenWorms" "Epilepsy" "ERing" "EthanolConcentration")
# UEA_MTSC30=("FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat")
# UEA_MTSC30=("InsectWingbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery")
# UEA_MTSC30=("NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports")
# UEA_MTSC30=("StandWalkJump" "UWaveGestureLibrary" "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits")
# UEA_MTSC30=("EigenWorms")

UEA_MTSC30=()
model="DLinear_CLS"
for dataset in ${UEA_MTSC30[@]}
do
    echo "Running ./scripts_classification/scripts_baseline/${model}_${dataset}.sh"
    echo "Result will be saved in ./scripts_classification/results/${model}_${dataset}.out"
    nohup bash ./scripts_classification/scripts_baseline/${model}_${dataset}.sh > ./scripts_classification/results/${model}_${dataset}.out &
done