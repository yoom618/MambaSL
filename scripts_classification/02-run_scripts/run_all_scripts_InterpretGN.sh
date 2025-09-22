# UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket")
# UEA_MTSC30=( "Epilepsy" "ERing" "EthanolConcentration")
# UEA_MTSC30=("FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat")
# UEA_MTSC30=("InsectWingbeat" "JapaneseVowels" "Libras" "LSST")
# UEA_MTSC30=("NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports")
# UEA_MTSC30=("SelfRegulationSCP1"  "SpokenArabicDigits" "UWaveGestureLibrary")
# UEA_MTSC30=("DuckDuckGeese" "EigenWorms")
# UEA_MTSC30=("MotorImagery")

UEA_MTSC30=()
model="InterpretGN_CLS"
for dataset in ${UEA_MTSC30[@]}
do
    echo "Running ./scripts_classification/scripts_baseline/${model}_${dataset}.sh"
    echo "Result will be saved in ./scripts_classification/results/${model}_${dataset}.out"
    nohup bash ./scripts_classification/scripts_baseline/${model}_${dataset}.sh > ./scripts_classification/results/${model}_${dataset}.out &
done

