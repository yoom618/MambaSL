# UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket")
# UEA_MTSC30=("DuckDuckGeese" "EigenWorms" "Epilepsy" "ERing" "EthanolConcentration")
# UEA_MTSC30=("FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat")
# UEA_MTSC30=("InsectWingbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery")
# UEA_MTSC30=("NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports")
# UEA_MTSC30=("StandWalkJump" "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "UWaveGestureLibrary")

# UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket" \
#             "DuckDuckGeese" "EigenWorms" "Epilepsy" "ERing" "EthanolConcentration" \
#             "FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat" \
#             "InsectWingbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery" \
#             "NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports" \
#             "StandWalkJump" "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "UWaveGestureLibrary")
model="TSLANet_CLS"
for dataset in ${UEA_MTSC30[@]}
do
    mkdir -p ./_run_TSLANet/results/${dataset}
    echo "Running ./scripts_classification/scripts_baseline/${model}_${dataset}.sh"
    echo "Result will be saved in ./_run_TSLANet/results directory"
    nohup bash ./scripts_classification/scripts_baseline/${model}_${dataset}.sh 1> /dev/null 2>&1 &
done