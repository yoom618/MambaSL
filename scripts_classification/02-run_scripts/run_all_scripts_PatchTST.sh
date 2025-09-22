# UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket")
# UEA_MTSC30=("EigenWorms" "Epilepsy" "ERing" "EthanolConcentration" "FingerMovements" "FaceDetection")
# UEA_MTSC30=("HandMovementDirection" "Handwriting" "InsectWingbeat" "PhonemeSpectra" "SpokenArabicDigits")
# UEA_MTSC30=("DuckDuckGeese")  # GPU 2
# UEA_MTSC30=("Heartbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery" "NATOPS")
# UEA_MTSC30=("PEMS-SF")  # GPU 2
# UEA_MTSC30=("PenDigits" "RacketSports" "SelfRegulationSCP1" "SelfRegulationSCP2" "StandWalkJump" "UWaveGestureLibrary")


UEA_MTSC30=()
model="PatchTST_CLS"
for dataset in ${UEA_MTSC30[@]}
do
    echo "Running ./scripts_classification/scripts_baseline/${model}_${dataset}.sh"
    echo "Result will be saved in ./scripts_classification/results/${model}_${dataset}.out"
    nohup bash ./scripts_classification/scripts_baseline/${model}_${dataset}.sh > ./scripts_classification/results/${model}_${dataset}.out &
done

