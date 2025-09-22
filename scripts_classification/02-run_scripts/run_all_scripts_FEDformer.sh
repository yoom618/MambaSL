# UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket")
# UEA_MTSC30=("DuckDuckGeese" "Epilepsy" "ERing" "EthanolConcentration" "FaceDetection" "FingerMovements")
# UEA_MTSC30=("HandMovementDirection" "Handwriting" "Heartbeat" "InsectWingbeat" "JapaneseVowels" "Libras") # GPU 1
# UEA_MTSC30=("LSST" "MotorImagery" "NATOPS" "PenDigits" "PEMS-SF"  "PhonemeSpectra")
# UEA_MTSC30=("RacketSports" "UWaveGestureLibrary" "SelfRegulationSCP1" "StandWalkJump" "SpokenArabicDigits")
# UEA_MTSC30=("EigenWorms") # GPU 3

UEA_MTSC30=()
model="FEDformer_CLS"
for dataset in ${UEA_MTSC30[@]}
do
    echo "Running ./scripts_classification/scripts_baseline/${model}_${dataset}.sh"
    echo "Result will be saved in ./scripts_classification/results/${model}_${dataset}.out"
    nohup bash ./scripts_classification/scripts_baseline/${model}_${dataset}.sh > ./scripts_classification/results/${model}_${dataset}.out &
done

